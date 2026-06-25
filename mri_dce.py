"""
mri_dce.py - 전임상 DCE-MRI / BBBO 분석 (단일 파일: 배치 CLI + 인터랙티브 GUI)

실행 모드:
  python mri_dce.py                      -> GUI (창에서 Load DICOM)
  python mri_dce.py --gui <dicom_dir>    -> GUI (폴더 바로 로드)
  python mri_dce.py <dicom_dir> <out>    -> 배치: 슬랩정리 + CSV + 곡선/해부/히트맵 PNG

라이브러리:
  from mri_dce import Study, enhancement_curve, bbbo_metrics, export_study
  study = Study.from_dicom_dir(r"C:\\path")
  for r in study.describe(): print(r)
  export_study(study, r"C:\\out")
  # ROI 정량: m = bbbo_metrics(slab, np.load("bbbo.npy"), np.load("contra.npy"))

의존성: numpy, pydicom, matplotlib (+scipy 선택, +SimpleITK 정합 선택, +tkinter GUI용)

핵심 규칙(이 데이터에서 직접 검증):
- 익명 DICOM은 태그가 비어 메타로 분류 불가 -> 기하구조(spacing/dims/orientation/origin)로 그룹핑.
- Volume=한 SeriesInstanceUID(한 시점), Slab=같은 기하구조 Volume들의 시간순(pre + posts).
- auto_roi는 탐색용만. 정량은 수동 ROI. 양측>50% 증강이면 ROI가 뇌실질 아님(정맥동/근육).
- 정합(registration): 시점마다 별개 시리즈라 움직임 있음 → (post-pre) 빼기 전 각 post를
  pre에 rigid 정합(SimpleITK, Mattes MI). 안 하면 경계에 움직임 인공물이 누출로 오인됨.
  CLI: --register / --reg,  GUI: '보기' 탭의 '정합' 체크박스,  라이브러리: study.register().
"""
from __future__ import annotations
import os
import re
import sys
import csv
from dataclasses import dataclass, field, replace
import numpy as np
import pydicom

try:
    from scipy import ndimage
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    import SimpleITK as sitk
    _HAS_SITK = True
except Exception:
    _HAS_SITK = False

# ---- 백엔드: 항상 Agg. GUI는 FigureCanvasTkAgg를 직접 임포트해 쓴다(DCEApp 참조).
#      pyplot 저장 함수가 GUI 실행 중에도 창을 안 띄우게 하기 위함.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _set_korean_font():
    """matplotlib에 한글 폰트 등록. 없으면 □□ 깨짐 → Windows 'Malgun Gothic' 우선."""
    from matplotlib import font_manager as _fm
    avail = {f.name for f in _fm.fontManager.ttflist}
    for name in ("Malgun Gothic", "NanumGothic", "NanumBarunGothic",
                 "AppleGothic", "Noto Sans CJK KR", "Gulim", "Batang"):
        if name in avail:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False   # 한글폰트는 마이너스 글리프 없음 → ASCII '-'


_set_korean_font()

_AXIS = {0: "sagittal", 1: "coronal", 2: "axial"}
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ============================================================ 유틸
def _seconds(t):
    if t is None:
        return None
    s = str(t).strip().split(".")[0]
    if not s:
        return None
    s = s.zfill(6)[:6]
    try:
        return int(s[0:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:6])
    except ValueError:
        return None


def _tp_rank(path):
    """파일/폴더 경로에서 시점 라벨 추출. PRE→0, POSTn→n, 없으면 None."""
    mt = re.search(r'(PRE|POST)\s*_?\s*(\d*)', path, re.I)
    if not mt:
        return None
    if mt.group(1).upper() == "PRE":
        return 0
    return int(mt.group(2)) if mt.group(2) else 1


def classify_orientation(iop) -> str:
    r, c = np.array(iop[:3], float), np.array(iop[3:], float)
    return _AXIS[int(np.argmax(np.abs(np.cross(r, c))))]


# ============================================================ 데이터 모델
@dataclass
class Volume:
    data: np.ndarray            # [z, y, x] float32 (rescale 적용)
    spacing: tuple              # (sx, sy, sz) mm
    origin: tuple
    iop: tuple
    orientation: str
    series_uid: str
    acq_time: float | None
    order: int
    condition: str = ""
    tp: int | None = None       # PRE=0, POSTn=n (파일/폴더명에서 파싱; 정렬 최우선)

    @property
    def n_slices(self): return self.data.shape[0]

    def signature(self, origin_round=0.5):
        def rnd(v):
            return round(round(v / origin_round) * origin_round, 3)
        return (round(self.spacing[0], 3), self.data.shape[0], self.orientation,
                rnd(self.origin[0]), rnd(self.origin[1]), rnd(self.origin[2]))

    def sort_key(self):
        # PRE/POST 라벨이 있으면 그게 최우선(촬영 순서 그라운드 트루스),
        # 없으면 acq_time, 그 다음 walk 순서.
        return (self.tp if self.tp is not None else 1 << 20,
                self.acq_time if self.acq_time is not None else 1e18, self.order)


@dataclass
class Slab:
    volumes: list                # list[Volume], index 0 = pre-Gd
    orientation: str
    animal: str = "?"

    @property
    def pre(self): return self.volumes[0]
    @property
    def n_frames(self): return len(self.volumes)
    @property
    def n_slices(self): return self.pre.n_slices
    @property
    def spacing(self): return self.pre.spacing

    @property
    def voxel_mm3(self):
        sx, sy, sz = self.spacing
        return abs(sx * sy * sz)

    @property
    def stack(self): return np.stack([v.data for v in self.volumes], axis=0)

    @property
    def times_min(self):
        t0 = self.pre.acq_time
        if t0 is None:
            return [None] * self.n_frames
        return [round((v.acq_time - t0) / 60.0, 2) if v.acq_time is not None else None
                for v in self.volumes]

    def label(self): return f"{self.animal}_{self.orientation}"


@dataclass
class Study:
    slabs: list = field(default_factory=list)

    @classmethod
    def from_dicom_dir(cls, path, origin_round=0.5, recursive=True) -> "Study":
        vols = _load_volumes(path, recursive)
        st = cls(slabs=_group_slabs(vols, origin_round))
        st.auto_group_animals()
        return st

    def auto_group_animals(self) -> None:
        conds = {s.pre.condition for s in self.slabs}
        if "" not in conds and len(conds) >= 2:
            for s in self.slabs:
                s.animal = s.pre.condition
            return
        order = sorted(self.slabs, key=lambda s: s.pre.sort_key())
        px = sorted({round(s.spacing[0], 3) for s in order})
        # 픽셀 크기로 종 판별: 쥐(mouse)≈0.078mm / 랫(rat)≈0.117mm → 0.1mm 경계
        species = {p: ("Rat" if p >= 0.1 else "Mouse") for p in px}
        seen, count = {}, {}
        for s in order:
            sp = species[round(s.spacing[0], 3)]
            if s.orientation in seen.get(sp, set()):
                count[sp] = count.get(sp, 1) + 1
                seen[sp] = set()
            else:
                count.setdefault(sp, 1)
            seen.setdefault(sp, set()).add(s.orientation)
            s.animal = f"{sp}{count[sp]}"

    def register(self, mode="inplane", guard=True, verbose=True):
        """모든 슬랩을 pre 기준 rigid 정합(제자리 치환 후 self 반환).
        (post-pre) 빼기 전 단계 — 움직임을 누출로 오인하지 않게 함.
        mode='inplane'(기본): 다중슬라이스에서 안전. 'rigid3d': 6자유도.
        guard=True(기본): 정렬이 실제로 개선된 post만 정합 채택(안 되면 원본 유지)."""
        out = []
        for s in self.slabs:
            if verbose:
                print(f"  정합 {s.label()} (post {s.n_frames - 1}개, mode={mode}, guard={guard})...")
            out.append(register_slab(s, mode, guard, verbose))
        self.slabs = out
        return self

    def set_animals(self, mapping):
        for i, s in enumerate(self.slabs):
            if i in mapping:
                s.animal = mapping[i]

    def animals(self):
        out = {}
        for s in self.slabs:
            out.setdefault(s.animal, []).append(s)
        return out

    def describe(self):
        rows = []
        for s in sorted(self.slabs, key=lambda s: (s.animal, s.orientation)):
            rows.append({"animal": s.animal, "orientation": s.orientation,
                         "frames": s.n_frames, "px_mm": round(s.spacing[0], 4),
                         "fov_mm": round(s.spacing[0] * s.pre.data.shape[1], 1),
                         "n_slices": s.pre.n_slices,
                         "origin": tuple(round(o, 1) for o in s.pre.origin)})
        return rows


# ============================================================ 로딩
def _load_volumes(path, recursive):
    if recursive:
        files = []
        for root, _, fs in os.walk(path):
            files += [os.path.join(root, f) for f in fs]
    else:
        files = [os.path.join(path, f) for f in os.listdir(path)]
    byseries = {}
    for order, fp in enumerate(files):
        try:
            ds = pydicom.dcmread(fp, force=True)
            if not hasattr(ds, "PixelData"):
                continue
        except Exception:
            continue
        uid = getattr(ds, "SeriesInstanceUID", fp)
        cond = os.path.basename(os.path.dirname(os.path.dirname(fp)))
        byseries.setdefault(uid, []).append((order, ds, cond, fp))

    vols = []
    for uid, items in byseries.items():
        items.sort(key=lambda x: x[0])
        ds0, cond, fp0 = items[0][1], items[0][2], items[0][3]
        iop = [float(x) for x in getattr(ds0, "ImageOrientationPatient",
                                         [1, 0, 0, 0, 1, 0])]
        n = np.cross(np.array(iop[:3]), np.array(iop[3:]))

        def proj(ds):
            ipp = np.array([float(x) for x in
                            getattr(ds, "ImagePositionPatient", [0, 0, 0])])
            return float(np.dot(ipp, n))
        uniq, seen = [], set()
        for d in sorted((it[1] for it in items), key=proj):
            k = round(proj(d), 3)
            if k in seen:
                continue
            seen.add(k); uniq.append(d)
        slope = float(getattr(ds0, "RescaleSlope", 1) or 1)
        inter = float(getattr(ds0, "RescaleIntercept", 0) or 0)
        arr = np.stack([d.pixel_array.astype(np.float32) * slope + inter for d in uniq])
        ps = getattr(ds0, "PixelSpacing", [1.0, 1.0])
        sy, sx = float(ps[0]), float(ps[1])
        sz = abs(proj(uniq[1]) - proj(uniq[0])) if len(uniq) > 1 \
            else float(getattr(ds0, "SliceThickness", 1.0) or 1.0)
        origin = tuple(float(x) for x in getattr(ds0, "ImagePositionPatient", [0, 0, 0]))
        acq = _seconds(getattr(ds0, "AcquisitionTime", None)) or \
            _seconds(getattr(ds0, "SeriesTime", None)) or \
            _seconds(getattr(ds0, "ContentTime", None)) or \
            _seconds(getattr(ds0, "InstanceCreationTime", None))
        tp = _tp_rank(fp0) if _tp_rank(fp0) is not None else _tp_rank(cond)
        vols.append(Volume(data=arr, spacing=(sx, sy, sz), origin=origin,
                           iop=tuple(iop), orientation=classify_orientation(iop),
                           series_uid=uid, acq_time=acq, order=items[0][0],
                           condition=cond, tp=tp))
    return vols


def _group_slabs(vols, origin_round):
    groups = {}
    for v in vols:
        groups.setdefault(v.signature(origin_round), []).append(v)
    slabs = []
    for sig, vs in groups.items():
        vs.sort(key=lambda v: v.sort_key())
        slabs.append(Slab(volumes=vs, orientation=vs[0].orientation))
    return slabs


# ============================================================ 정합 (registration)
def _sitk_image(vol):
    """Volume.data([z,y,x]) → SimpleITK float32 이미지. spacing (sx,sy,sz) 부여.
    슬랩 내 볼륨은 같은 기하구조 → 공통 합성 좌표계(origin 0, direction 단위)에서
    array-to-array 정합이면 충분(origin/orientation 태그는 이미 그룹핑에 소진됨)."""
    img = sitk.GetImageFromArray(np.ascontiguousarray(vol.data, dtype=np.float32))
    sx, sy, sz = vol.spacing
    img.SetSpacing((float(sx), float(sy), float(sz)))
    return sitk.Cast(img, sitk.sitkFloat32)


# SimpleITK GetImageFromArray는 array[z,y,x]를 항상 image z=슬라이스로 매핑(해부방향 무관).
# 따라서 '면외(through-plane)'는 어느 슬랩이든 image z. Euler3D 파라미터 순서:
#   (angleX, angleY, angleZ, tx, ty, tz)  → 면내(in-plane)만 = angleZ, tx, ty.
_INPLANE_WEIGHTS = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]   # 면외(angleX,angleY,tz) 동결


def _register_pair(fixed, moving, mode="inplane"):
    """moving을 fixed에 rigid(Euler3D) 정합해 변환 반환.
    Mattes MI라 조영 증강 자체엔 안 속음. 뇌는 두개골에 고정 → rigid로 충분(deformable 금지:
    측정하려는 증강을 왜곡함). 샘플링 seed 고정으로 재현성 확보.
    mode='inplane'(기본): 면내(tx,ty,면내회전)만 추정 — 두꺼운 다중슬라이스(적은 슬라이스 수)에서
        면외 자유도는 구속이 안 돼 오정합을 유발하므로 동결. 실측에서 3D는 axial을 망침.
    mode='rigid3d': 6자유도 전부 — 등방성/슬라이스 많은 데이터에서만."""
    init = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, 1234)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=1e-4, numberOfIterations=200,
        gradientMagnitudeTolerance=1e-8)
    reg.SetOptimizerScalesFromPhysicalShift()
    if mode == "inplane":
        reg.SetOptimizerWeights(_INPLANE_WEIGHTS)
    reg.SetInitialTransform(init, inPlace=False)
    return reg.Execute(fixed, moving)


def _grad_mag(v):
    g = np.gradient(v.astype(np.float32))
    return np.sqrt(sum(gi * gi for gi in g))


def _edge_ncc(a, b, mask):
    """엣지(gradient magnitude) 정규화상관. 움직임엔 민감, 부드러운 조영 증강엔 둔감 →
    '정합이 정렬을 실제로 개선했는가'의 판정 지표. 1=완벽 정렬."""
    ga, gb = _grad_mag(a)[mask], _grad_mag(b)[mask]
    ga = ga - ga.mean(); gb = gb - gb.mean()
    return float((ga * gb).sum() / (np.sqrt((ga ** 2).sum() * (gb ** 2).sum()) + 1e-9))


def register_slab(slab, mode="inplane", guard=True, verbose=False):
    """슬랩의 각 post 볼륨을 pre(volumes[0])에 rigid 정합한 새 Slab 반환(원본 불변).
    pre는 그대로 두고 post만 pre 격자로 리샘플 → 이후 (post-pre) 빼기가 해부학적으로 정합됨.
    반환 볼륨 shape는 pre와 동일하므로 기존 ROI 마스크와 그대로 호환.
    mode: 'inplane'(기본, 다중슬라이스 안전) | 'rigid3d'(6자유도).
    guard=True(기본): 엣지 정합도가 실제로 좋아진 post만 정합본 채택, 아니면 원본 유지.
        잘 고정된 동물(움직임 ~0)에선 정합이 보간 블러로 손해만 내므로 이 가드가 필수.
        실측: 이 가드로 axial(이미 정렬됨)은 원본 유지, coronal(움직임 있음)만 정합 채택."""
    if not _HAS_SITK:
        raise RuntimeError("정합에는 SimpleITK 필요: pip install SimpleITK")
    fixed = _sitk_image(slab.pre)
    pre = slab.pre.data
    tissue = pre > np.mean(pre[pre > 0])
    new_vols = [slab.volumes[0]]
    for i, v in enumerate(slab.volumes[1:], 1):
        moving = _sitk_image(v)
        tx = _register_pair(fixed, moving, mode)
        res = sitk.Resample(moving, fixed, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        arr = sitk.GetArrayFromImage(res).astype(np.float32)
        p = tx.GetParameters()   # (rx,ry,rz rad, tx,ty,tz mm)
        shift = float(np.hypot(p[3], np.hypot(p[4], p[5])))
        if guard:
            b, a = _edge_ncc(pre, v.data, tissue), _edge_ncc(pre, arr, tissue)
            if a <= b:                      # 정합이 정렬을 개선 못 함 → 원본 유지
                if verbose:
                    print(f"    post{i}: 정합 미개선(엣지 {b:.3f}→{a:.3f}) → 원본 유지")
                new_vols.append(v); continue
            if verbose:
                print(f"    post{i}: 채택(엣지 {b:.3f}→{a:.3f}) |t|={shift:.3f}mm")
            new_vols.append(replace(v, data=arr)); continue
        if verbose:
            print(f"    post{i}: 회전(rad)=({p[0]:+.4f},{p[1]:+.4f},{p[2]:+.4f}) "
                  f"이동(mm)=({p[3]:+.3f},{p[4]:+.3f},{p[5]:+.3f}) |t|={shift:.3f}")
        new_vols.append(replace(v, data=arr))
    return Slab(volumes=new_vols, orientation=slab.orientation, animal=slab.animal)


# ============================================================ DCE 정량
def _mask_or_all(slab, mask):
    if mask is None:
        m = slab.pre.data > np.mean(slab.pre.data[slab.pre.data > 0])
    else:
        m = np.asarray(mask, bool)
    if m.shape != slab.pre.data.shape:
        raise ValueError(f"mask shape {m.shape} != volume {slab.pre.data.shape}")
    return m


def enhancement_curve(slab, mask=None):
    m = _mask_or_all(slab, mask)
    sig = np.array([v.data[m].mean() for v in slab.volumes], float)
    base = sig[0] if sig[0] != 0 else 1.0
    pct = (sig - base) / base * 100.0
    times = slab.times_min
    tnum = [t if t is not None else i for i, t in enumerate(times)]
    peak_i = int(np.argmax(pct))
    dt = (tnum[1] - tnum[0]) if len(tnum) > 1 and tnum[1] != tnum[0] else 1.0
    washin = float(pct[1] / dt) if len(pct) > 1 else 0.0
    return {"label": slab.label(), "n_frames": slab.n_frames,
            "roi_voxels": int(m.sum()),
            "roi_mm3": round(float(m.sum()) * slab.voxel_mm3, 3),
            "times_min": times, "signal": [round(float(x), 1) for x in sig],
            "enh_pct": [round(float(x), 1) for x in pct],
            "peak_pct": round(float(pct[peak_i]), 1),
            "time_to_peak_frame": peak_i,
            "washin_pct_per_unit": round(washin, 1),
            "auc_pct": round(float(_trapz(pct, tnum)), 1)}


def bbbo_metrics(slab, roi_mask, control_mask, k_sd=2.0):
    roi = enhancement_curve(slab, roi_mask)
    ctl = enhancement_curve(slab, control_mask)
    net = [round(a - b, 1) for a, b in zip(roi["enh_pct"], ctl["enh_pct"])]
    m = np.asarray(roi_mask, bool)
    cc = np.asarray(control_mask, bool)
    pre = slab.pre.data
    peak_vol = slab.volumes[int(np.argmax(roi["enh_pct"]))].data
    peak_enh = (peak_vol - pre) / (pre + 1e-6) * 100.0
    vals = peak_enh[cc]
    thr = float(vals.mean() + k_sd * vals.std())
    opening = m & (peak_enh > thr)
    return {"label": slab.label(), "roi": roi, "control": ctl,
            "net_leakage_pct": net, "peak_net_leakage_pct": max(net),
            "opening_threshold_pct": round(thr, 1),
            "opening_voxels": int(opening.sum()),
            "opening_mm3": round(float(opening.sum()) * slab.voxel_mm3, 3),
            "verdict": "POSITIVE" if max(net) >= 15 else "NEGATIVE/equivocal"}


def auto_roi(slab, top_pct=90.0, min_voxels=50):
    """탐색용만 — Gd 배설/혈관을 오인하므로 정량 근거로 쓰지 말 것."""
    pre = slab.pre.data
    peak = np.maximum.reduce([v.data for v in slab.volumes[1:]])
    tissue = pre > np.mean(pre[pre > 0])
    E = (peak - pre) / (pre + 1e-6)
    mask = tissue & (E >= np.percentile(E[tissue], top_pct))
    if not _HAS_SCIPY:
        return mask
    lbl, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    roi = lbl == (int(np.argmax(sizes)) + 1)
    return roi if roi.sum() >= min_voxels else mask


def enhancement_map(slab, frame="peak", clip=(0, 100)):
    pre = slab.pre.data
    post = slab.volumes[-1].data if frame == "peak" else slab.volumes[int(frame)].data
    E = np.where(pre > np.mean(pre[pre > 0]),
                 (post - pre) / (pre + 1e-6) * 100.0, np.nan)
    return np.clip(E, *clip)


# ============================================================ 시각화 (배치)
def _rep_slices(n, k=5):
    lo, hi = int(n * 0.2), max(int(n * 0.8), int(n * 0.2) + 1)
    return list(np.unique(np.linspace(lo, hi - 1, min(k, hi - lo)).round().astype(int)))


def _save(fig, out, dpi):
    if out:
        fig.savefig(out, dpi=dpi, bbox_inches="tight"); plt.close(fig); return out
    return fig


def montage_anatomy(slab, frame="pre", out=None, k=5, dpi=130):
    vol = slab.pre if frame == "pre" else \
        (slab.volumes[-1] if frame == "peak" else slab.volumes[int(frame)])
    a = vol.data
    idx = _rep_slices(a.shape[0], k)
    wl = np.percentile(a, [1, 99])
    fig, axes = plt.subplots(1, len(idx), figsize=(3 * len(idx), 3.2))
    axes = np.atleast_1d(axes)
    for ax, z in zip(axes, idx):
        ax.imshow(a[z], cmap="gray", vmin=wl[0], vmax=wl[1])
        ax.set_title(f"slice {z}", fontsize=9); ax.axis("off")
    fig.suptitle(f"{slab.label()}  ({frame}-Gd anatomy)", fontsize=12)
    fig.tight_layout()
    return _save(fig, out, dpi)


def montage_heatmap(slab, out=None, k=5, vmax=100, dpi=130):
    peak = slab.volumes[-1].data
    E = enhancement_map(slab, "peak", clip=(0, vmax))
    idx = _rep_slices(peak.shape[0], k)
    wl = np.percentile(peak, [1, 99])
    fig, axes = plt.subplots(1, len(idx), figsize=(3 * len(idx), 3.4))
    axes = np.atleast_1d(axes)
    im = None
    for ax, z in zip(axes, idx):
        ax.imshow(peak[z], cmap="gray", vmin=wl[0], vmax=wl[1])
        im = ax.imshow(E[z], cmap="jet", vmin=0, vmax=vmax, alpha=0.55)
        ax.set_title(f"slice {z}", fontsize=9); ax.axis("off")
    fig.suptitle(f"{slab.label()}  enhancement % (peak-pre)/pre", fontsize=12)
    fig.subplots_adjust(right=0.9)
    fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.015, 0.7]), label="% enhancement")
    return _save(fig, out, dpi)


def center_slice(slab, frame="pre", out=None, dpi=140):
    """단일 중앙 슬라이스 1장. frame: 'pre' | 'peak'/'post' | 정수 인덱스."""
    vol = slab.pre if frame == "pre" else \
        (slab.volumes[-1] if frame in ("peak", "post") else slab.volumes[int(frame)])
    a = vol.data
    z = a.shape[0] // 2
    wl = np.percentile(a, [1, 99])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(a[z], cmap="gray", vmin=wl[0], vmax=wl[1])
    ax.set_title(f"{slab.label()}  ({frame}-Gd, slice {z})", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    return _save(fig, out, dpi)


def save_frame_slices(slab, z=None, outdir=".", wl=None, dpi=140):
    """슬라이스 z에서 pre 1장 + post 각 시점 1장씩(개별, 무축·무제목 bare). 고정 윈도우라 밝기차=실제 신호차."""
    a0 = slab.volumes[0].data
    z = a0.shape[0] // 2 if z is None else int(z)
    if wl is None:
        wl = np.percentile(slab.volumes[-1].data, [1, 99])
    tag = slab.label().replace(" ", "_"); saved = []

    def _bare(arr, name):
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(arr, cmap="gray", vmin=wl[0], vmax=wl[1]); ax.axis("off")
        p = os.path.join(outdir, name)
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0); plt.close(fig)
        saved.append(p)

    _bare(slab.volumes[0].data[z], f"{tag}_slice{z}_preGd.png")
    for fi in range(1, slab.n_frames):
        _bare(slab.volumes[fi].data[z], f"{tag}_slice{z}_post{fi}.png")
    return saved


def heatmap_frames_row(slab, z=None, out=None, vmin=0, vmax=100, dpi=130):
    """슬라이스 z에서 post 각 시점 enhancement 히트맵을 한 줄(일렬)로. 시점별 (post-pre)/pre %.
    vmin~vmax = 컬러 스케일(색 포화는 clim이 처리; GUI scale 입력칸과 동일 의미)."""
    pre = slab.volumes[0].data
    z = pre.shape[0] // 2 if z is None else int(z)
    pre_z = pre[z].astype(float)
    thr = float(np.mean(pre_z[pre_z > 0])) if np.any(pre_z > 0) else 0.0
    wl = np.percentile(slab.volumes[-1].data, [1, 99])
    n = slab.n_frames - 1
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.4))
    axes = np.atleast_1d(axes); im = None
    for k, fi in enumerate(range(1, slab.n_frames)):
        post = slab.volumes[fi].data[z].astype(float)
        E = np.where(pre_z > thr, (post - pre_z) / (pre_z + 1e-6) * 100.0, np.nan)
        axes[k].imshow(post, cmap="gray", vmin=wl[0], vmax=wl[1])
        im = axes[k].imshow(E, cmap="jet", vmin=vmin, vmax=vmax, alpha=0.55)
        axes[k].set_title(f"post {fi}", fontsize=10); axes[k].axis("off")
    fig.suptitle(f"{slab.orientation}  slice {z}  enhancement % (post-pre)/pre", fontsize=12)
    fig.subplots_adjust(right=0.9)
    fig.colorbar(im, cax=fig.add_axes([0.92, 0.18, 0.012, 0.64]), label="% enh")
    return _save(fig, out, dpi)


def subtraction_frames_row(slab, z=None, out=None, dpi=130):
    """슬라이스 z에서 post 각 시점 차감(ΔS=post-pre)을 한 줄로. 발산 컬러맵(빨강=증가/파랑=감소).
    %증강과 달리 감소까지 부호 그대로 보여줌."""
    pre = slab.volumes[0].data
    z = pre.shape[0] // 2 if z is None else int(z)
    pre_z = pre[z].astype(float)
    M = float(np.percentile(np.abs(slab.volumes[-1].data[z].astype(float) - pre_z), 99)) or 1.0
    n = slab.n_frames - 1
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3.4))
    axes = np.atleast_1d(axes); im = None
    for k, fi in enumerate(range(1, slab.n_frames)):
        d = slab.volumes[fi].data[z].astype(float) - pre_z
        im = axes[k].imshow(d, cmap="seismic", vmin=-M, vmax=M)
        axes[k].set_title(f"POST{fi}-PRE", fontsize=10); axes[k].axis("off")
    fig.suptitle(f"{slab.orientation}  slice {z}  subtraction ΔS (post-pre)", fontsize=12)
    fig.subplots_adjust(right=0.9)
    fig.colorbar(im, cax=fig.add_axes([0.92, 0.18, 0.012, 0.64]), label="ΔS (red=증가)")
    return _save(fig, out, dpi)


def plot_curves(curves, out=None, dpi=130):
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in curves:
        x = [t if t is not None else i for i, t in enumerate(c["times_min"])]
        ax.plot(x, c["enh_pct"], marker="o", label=c["label"])
    ax.set_xlabel("time (min or frame)"); ax.set_ylabel("% enhancement")
    ax.axhline(0, color="k", lw=0.5); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, out, dpi)


def export_study(study, outdir, frames=("pre", "peak"), heatmap=True):
    os.makedirs(outdir, exist_ok=True)
    saved = []
    for s in study.slabs:
        for fr in frames:
            p = os.path.join(outdir, f"{s.label()}_{fr}Gd_anatomy.png")
            montage_anatomy(s, fr, p); saved.append(os.path.basename(p))
        if heatmap:
            p = os.path.join(outdir, f"{s.label()}_heatmap.png")
            montage_heatmap(s, p); saved.append(os.path.basename(p))
    return saved


def run_batch(dicom_dir, out_dir, register=False):
    os.makedirs(out_dir, exist_ok=True)
    study = Study.from_dicom_dir(dicom_dir)
    if register:
        print("[정합 rigid: 각 post → pre]")
        study.register()
    print(f"\n[슬랩 {len(study.slabs)}개 / 동물 {len(study.animals())}마리]")
    for r in study.describe():
        print("  ", r)
    curves = [enhancement_curve(s) for s in study.slabs]
    csv_path = os.path.join(out_dir, "enhancement_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["label", "orientation", "n_frames", "roi_mm3", "peak_pct",
                    "time_to_peak_frame", "washin", "auc"])
        for s, c in zip(study.slabs, curves):
            w.writerow([c["label"], s.orientation, c["n_frames"], c["roi_mm3"],
                        c["peak_pct"], c["time_to_peak_frame"],
                        c["washin_pct_per_unit"], c["auc_pct"]])
    plot_curves(curves, os.path.join(out_dir, "all_curves.png"))
    saved = export_study(study, out_dir)
    print(f"\nCSV -> {csv_path}\nPNG {len(saved)}개 -> {out_dir}")


# ============================================================ GUI 마스크
def _disk_mask(shape, sl, cx, cy, r):
    z, y, x = shape
    m = np.zeros(shape, bool)
    yy, xx = np.ogrid[:y, :x]
    m[int(sl)] = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    return m


def _poly_mask(shape, sl, verts):
    """자유곡선(폴리곤) 마스크. verts = [(x,y), ...]."""
    from matplotlib.path import Path
    z, y, x = shape
    m = np.zeros(shape, bool)
    if len(verts) < 3:
        return m
    yy, xx = np.mgrid[:y, :x]
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    m[int(sl)] = Path(np.asarray(verts)).contains_points(pts).reshape(y, x)
    return m


# ============================================================ GUI (tkinter)
class DCEApp:
    """tkinter 네이티브 위젯(슬라이더/버튼) + matplotlib는 이미지/곡선 캔버스로만.
    이미지 캔버스와 곡선 캔버스를 분리 → frame/slice 스크럽이 곡선을 재계산하지 않음(멈춤 방지).
    ROI 드래그는 blitting으로 외곽선만 갱신."""

    def __init__(self, path=None, register=False):
        import tkinter as tk
        from tkinter import ttk, filedialog
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.patches import Circle
        self._tk, self._ttk, self._fd, self._Circle = tk, ttk, filedialog, Circle

        # ---- 상태 ----
        self.slabs, self.si = [], 0
        self._auto_reg = bool(register)       # 로드 직후 자동 정합할지(--reg)
        self._slabs_raw = []                  # 원본(정합 전) — 토글 복귀용
        self._slabs_reg = None                # 정합 결과 캐시(1회 계산 후 재사용)
        self.slice = self.frame = 0
        self.mid = 0.0
        self.heat = False
        self.sub = False                # 차감뷰(ΔS=post-pre, 발산컬러맵)
        self.hm_vmin, self.hm_vmax = 0.0, 100.0   # 히트맵 컬러 스케일(% enhancement)
        self.mode = "circle"            # 'circle' | 'free'
        self.contra_mode = "mirror"     # 'mirror' | 'manual'
        self.target = "roi"             # 'roi'(누출부) | 'roi2'(대조)
        self.roi = self.roi2 = None
        self._press = None; self._poly = []; self._last = None
        self._im_post = None; self._roi_artists = []
        self._bg = None; self._drag_artist = None; self._cur = None; self._r2 = False
        self._wl = None; self._pre_thr = None; self._sub_M = None

        # ---- 윈도우 ----
        self.root = tk.Tk()
        self.root.title("DCE-MRI / BBBO ROI 분석")
        self.root.geometry("1320x820")
        try:
            self.root.option_add("*Font", ("Malgun Gothic", 10))
        except Exception:
            pass

        top = ttk.Frame(self.root); top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(8, 2))
        ttk.Label(top, text="DCE-MRI · BBBO ROI 분석",
                  font=("Malgun Gothic", 14, "bold")).pack(side=tk.LEFT)
        self.lbl_sub = ttk.Label(top, text="Load DICOM 으로 시작", foreground="#666")
        self.lbl_sub.pack(side=tk.LEFT, padx=14)

        # ---- 캔버스 2개(이미지 | 곡선) ----
        mid = ttk.Frame(self.root); mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.fig_img = Figure(figsize=(7.4, 4.6), facecolor="#f4f4f2")
        self.ax_pre = self.fig_img.add_axes([0.01, 0.04, 0.47, 0.90])
        self.ax_img = self.fig_img.add_axes([0.51, 0.04, 0.47, 0.90])
        for ax in (self.ax_pre, self.ax_img):
            ax.set_facecolor("black"); ax.axis("off")
        self.ax_img.set_title("Load DICOM 클릭", fontsize=11, color="#555")
        self.canvas_img = FigureCanvasTkAgg(self.fig_img, master=mid)
        self.canvas_img.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig_cur = Figure(figsize=(5.0, 4.6), facecolor="#f4f4f2")
        self.ax_cur = self.fig_cur.add_axes([0.14, 0.12, 0.82, 0.82])
        self.ax_cur.grid(alpha=0.3)
        self.canvas_cur = FigureCanvasTkAgg(self.fig_cur, master=mid)
        self.canvas_cur.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ---- 슬라이더 ----
        ctr = ttk.Frame(self.root); ctr.pack(side=tk.TOP, fill=tk.X, padx=14, pady=2)
        self.s_slice, self.lab_slice = self._mk_scale(ctr, "slice", self._on_slice)
        self.s_frame, self.lab_frame = self._mk_scale(ctr, "frame", self._on_frame)
        self.s_mid, self.lab_mid = self._mk_scale(ctr, "midline x", self._on_mid)

        # ---- 탭 컨트롤(노트북): 한 창에서 파일/보기/ROI/저장 전환 ----
        nb = ttk.Notebook(self.root)
        nb.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(4, 10))

        # 탭1: 파일 · 정보 (Load 밑에 DICOM 픽셀 정보)
        tab_file = ttk.Frame(nb); nb.add(tab_file, text="파일 · 정보")
        row1 = ttk.Frame(tab_file); row1.pack(side=tk.TOP, fill=tk.X, pady=(6, 2))
        ttk.Button(row1, text="Load DICOM", command=self._load).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="◀ slab", command=lambda: self._slab(-1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(row1, text="slab ▶", command=lambda: self._slab(1)).pack(side=tk.LEFT, padx=2)
        info_fr = ttk.Frame(tab_file)
        info_fr.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=2, pady=(2, 6))
        self.txt_info = tk.Text(info_fr, height=7, font=("Malgun Gothic", 9), wrap="none",
                                background="#1b1b1b", foreground="#d8d8d8", borderwidth=0)
        sb = ttk.Scrollbar(info_fr, orient="vertical", command=self.txt_info.yview)
        self.txt_info.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.txt_info.insert("1.0", "DICOM 불러오면 픽셀 정보 표시")
        self.txt_info.config(state="disabled")

        # 탭2: 보기 (heatmap / 차감 / 스케일)
        tab_view = ttk.Frame(nb); nb.add(tab_view, text="보기")
        vrow = ttk.Frame(tab_view); vrow.pack(side=tk.TOP, fill=tk.X, pady=8)
        self.var_heat = tk.BooleanVar(value=False)
        ttk.Checkbutton(vrow, text="heatmap", variable=self.var_heat,
                        command=self._toggle_heat).pack(side=tk.LEFT, padx=(6, 2))
        self.var_sub = tk.BooleanVar(value=False)
        ttk.Checkbutton(vrow, text="차감ΔS", variable=self.var_sub,
                        command=self._toggle_sub).pack(side=tk.LEFT, padx=2)
        self.var_reg = tk.BooleanVar(value=False)
        ttk.Checkbutton(vrow, text="정합", variable=self.var_reg,
                        command=self._toggle_register).pack(side=tk.LEFT, padx=(10, 2))
        ttk.Label(vrow, text="│ scale").pack(side=tk.LEFT, padx=(10, 0))
        self.var_vmin = tk.StringVar(value="0")
        self.var_vmax = tk.StringVar(value="100")
        e_vmin = ttk.Entry(vrow, textvariable=self.var_vmin, width=5); e_vmin.pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(vrow, text="~").pack(side=tk.LEFT)
        e_vmax = ttk.Entry(vrow, textvariable=self.var_vmax, width=5); e_vmax.pack(side=tk.LEFT)
        for e in (e_vmin, e_vmax):
            e.bind("<Return>", lambda ev: self._apply_hmscale())
            e.bind("<FocusOut>", lambda ev: self._apply_hmscale())
        ttk.Label(vrow, text="% enhancement", foreground="#888").pack(side=tk.LEFT, padx=6)

        # 탭3: ROI (모양 / 대조 / 그릴곳 / clear)
        tab_roi = ttk.Frame(nb); nb.add(tab_roi, text="ROI")
        rrow = ttk.Frame(tab_roi); rrow.pack(side=tk.TOP, fill=tk.X, pady=8)
        ttk.Label(rrow, text="모양").pack(side=tk.LEFT, padx=(6, 2))
        self.var_mode = tk.StringVar(value="circle")
        for t, v in (("circle", "circle"), ("free", "free")):
            ttk.Radiobutton(rrow, text=t, variable=self.var_mode, value=v,
                            command=self._set_mode).pack(side=tk.LEFT)
        ttk.Label(rrow, text="│ 대조").pack(side=tk.LEFT, padx=(8, 0))
        self.var_contra = tk.StringVar(value="mirror")
        for t, v in (("미러", "mirror"), ("수동", "manual")):
            ttk.Radiobutton(rrow, text=t, variable=self.var_contra, value=v,
                            command=self._set_contra).pack(side=tk.LEFT)
        ttk.Label(rrow, text="│ 그릴곳").pack(side=tk.LEFT, padx=(8, 0))
        self.var_target = tk.StringVar(value="roi")
        for t, v in (("누출부", "roi"), ("대조", "roi2")):
            ttk.Radiobutton(rrow, text=t, variable=self.var_target, value=v,
                            command=self._set_target).pack(side=tk.LEFT)
        ttk.Button(rrow, text="clear ROI", command=self._clear).pack(side=tk.LEFT, padx=(12, 2))

        # 탭4: 저장
        tab_save = ttk.Frame(nb); nb.add(tab_save, text="저장")
        srow = ttk.Frame(tab_save); srow.pack(side=tk.TOP, fill=tk.X, pady=8)
        ttk.Button(srow, text="save PNG", command=self._save_png).pack(side=tk.LEFT, padx=4)
        ttk.Button(srow, text="save graph", command=self._save_curve).pack(side=tk.LEFT, padx=4)
        ttk.Button(srow, text="save 전체슬랩", command=self._save_all).pack(side=tk.LEFT, padx=4)

        # ---- 마우스 이벤트(이미지 캔버스에서만 ROI) ----
        self.canvas_img.mpl_connect("button_press_event", self._on_press)
        self.canvas_img.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas_img.mpl_connect("button_release_event", self._on_release)

        self.canvas_img.draw(); self.canvas_cur.draw()
        if path:
            self.root.after(120, lambda: self._load_path(path))

    def _mk_scale(self, parent, name, cb):
        tk, ttk = self._tk, self._ttk
        fr = ttk.Frame(parent); fr.pack(side=tk.TOP, fill=tk.X, pady=1)
        ttk.Label(fr, text=name, width=10).pack(side=tk.LEFT)
        s = ttk.Scale(fr, from_=0, to=1, orient="horizontal", command=cb)
        s.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        lab = ttk.Label(fr, text="0", width=5); lab.pack(side=tk.LEFT)
        return s, lab

    # ---------- 로딩 ----------
    def _ask_dir(self, title):
        return self._fd.askdirectory(title=title, parent=self.root) or None

    def _load(self):
        d = self._ask_dir("DICOM 폴더 선택")
        if d:
            self._load_path(d)

    def _load_path(self, d):
        self.lbl_sub.config(text="로딩 중..."); self.root.update_idletasks()
        st = Study.from_dicom_dir(d)
        self.slabs = sorted(st.slabs, key=lambda s: (s.animal, s.orientation))
        if not self.slabs:
            self.lbl_sub.config(text="DICOM 못 찾음"); return
        self._slabs_raw = self.slabs        # 원본 보관
        self._slabs_reg = None              # 새 데이터 → 정합 캐시 무효화
        self.var_reg.set(False)
        print(f"[{len(self.slabs)}슬랩 / {len(st.animals())}마리]")
        for s in self.slabs:
            print(f"  {s.label():28s} frames={s.n_frames} slices={s.n_slices}")
        self.si = 0; self._setup_slab()
        if self._auto_reg and _HAS_SITK:    # --reg 로 켜고 시작한 경우 자동 정합
            self.var_reg.set(True); self._toggle_register()

    def _setup_slab(self):
        s = self.slabs[self.si]
        z, y, x = s.pre.data.shape
        self.slice, self.frame = z // 2, s.n_frames - 1   # 기본=peak post(증강 보임)
        self.mid = x / 2.0
        self.roi = self.roi2 = None
        self._wl = np.percentile(s.volumes[-1].data, [1, 99])
        pre = s.volumes[0].data
        self._pre_thr = float(np.mean(pre[pre > 0]))   # 조직 임계(저신호 ÷폭발 배제)
        self._sub_M = float(np.percentile(np.abs(s.volumes[-1].data - pre), 99)) or 1.0  # 차감 대칭 스케일
        self._init_artists(s)
        self.s_slice.config(to=z - 1); self.s_slice.set(self.slice)
        self.s_frame.config(from_=1, to=s.n_frames - 1); self.s_frame.set(self.frame)  # post만(pre=왼쪽패널)
        self.s_mid.config(to=x - 1); self.s_mid.set(int(self.mid))
        self.lab_slice.config(text=str(self.slice))
        self.lab_frame.config(text=str(self.frame))
        self.lab_mid.config(text=str(int(self.mid)))
        self._fill_info()
        self._draw_image(); self._update_curve()

    def _fill_info(self):
        """'파일 · 정보' 탭에 슬랩별 DICOM 픽셀/매트릭스 정보 표시(현재 슬랩 표시)."""
        if not hasattr(self, "txt_info"):
            return
        t = self.txt_info
        t.config(state="normal"); t.delete("1.0", "end")
        if not self.slabs:
            t.insert("1.0", "DICOM 불러오면 픽셀 정보 표시"); t.config(state="disabled"); return
        lines = []
        for i, s in enumerate(self.slabs):
            sx, sy, sz = s.spacing
            z, y, x = s.pre.data.shape
            cur = "  ◀ 현재" if i == self.si else ""
            lines.append(f"[{i}] {s.animal} · {s.orientation}{cur}")
            lines.append(f"     픽셀 {sx:.4f} x {sy:.4f} mm | 두께 {abs(sz):.3f} mm | 복셀 {s.voxel_mm3:.5f} mm^3")
            lines.append(f"     매트릭스 {x} x {y} (x,y) | 슬라이스 {z} | FOV {sx*x:.1f} x {sy*y:.1f} mm")
            lines.append(f"     프레임 {s.n_frames} (pre + post{s.n_frames - 1})")
            lines.append("")
        t.insert("1.0", "\n".join(lines))
        t.config(state="disabled")

    def _slab(self, d):
        if self.slabs:
            self.si = (self.si + d) % len(self.slabs); self._setup_slab()

    def _toggle_register(self):
        """'정합' 체크: 켜면 pre 기준 rigid 정합 결과로, 끄면 원본으로 전환.
        정합은 1회만 계산해 캐시(_slabs_reg) → 껐다 켜도 재계산 안 함."""
        if not self._slabs_raw:
            self.var_reg.set(False); return
        if not _HAS_SITK:
            self.var_reg.set(False)
            self.lbl_sub.config(text="정합 불가: pip install SimpleITK 필요"); return
        if self.var_reg.get():
            if self._slabs_reg is None:
                self.lbl_sub.config(text="정합 중... (rigid, 시점 많으면 수십 초)")
                self.root.update_idletasks()
                try:
                    self._slabs_reg = [register_slab(s) for s in self._slabs_raw]
                except Exception as ex:
                    self.var_reg.set(False)
                    self.lbl_sub.config(text=f"정합 실패: {ex}"); return
            self.slabs = self._slabs_reg
        else:
            self.slabs = self._slabs_raw
        self.si = min(self.si, len(self.slabs) - 1)
        self._setup_slab()
        self.lbl_sub.config(text=("정합 적용됨(rigid, 각 post→pre)"
                                  if self.var_reg.get() else "정합 해제(원본)"))

    # ---------- 슬라이더 콜백 ----------
    def _on_slice(self, v):
        self.slice = int(float(v)); self.lab_slice.config(text=str(self.slice))
        self._draw_image()

    def _on_frame(self, v):
        self.frame = int(float(v)); self.lab_frame.config(text=str(self.frame))
        self._draw_image()

    def _on_mid(self, v):
        self.mid = float(v); self.lab_mid.config(text=str(int(self.mid)))
        self._draw_image()
        if self.contra_mode != "manual":   # 수동 ROI는 midline과 무관 → 곡선 재계산 불필요
            self._update_curve()

    # ---------- 이미지 ----------
    def _init_artists(self, s):
        """슬랩당 1번: 영구 이미지/미드라인 아티스트 생성. 이후엔 set_data로 갱신."""
        wl = self._wl
        z0 = s.volumes[0].data[self.slice]
        self.ax_pre.clear(); self.ax_img.clear()
        for ax in (self.ax_pre, self.ax_img):
            ax.set_facecolor("black"); ax.axis("off")
        self.ax_pre.set_title("pre-Gd", fontsize=10)
        self._im_pre = self.ax_pre.imshow(z0, cmap="gray", vmin=wl[0], vmax=wl[1])
        self._im_post = self.ax_img.imshow(s.volumes[self.frame].data[self.slice],
                                           cmap="gray", vmin=wl[0], vmax=wl[1])
        self._heat_im = self.ax_img.imshow(np.zeros_like(z0, float),
                                           cmap="jet", vmin=0, vmax=100, alpha=0.5)
        self._heat_im.set_visible(False)
        self._mid_pre = self.ax_pre.axvline(self.mid, color="y", lw=0.6, ls="--")
        self._mid_post = self.ax_img.axvline(self.mid, color="y", lw=0.6, ls="--")
        self._roi_artists = []

    def _draw_image(self):
        if not self.slabs:
            return
        s = self.slabs[self.si]
        self.slice = min(self.slice, s.n_slices - 1)
        self.frame = min(max(int(self.frame), 1), s.n_frames - 1)   # post1~ (frame 0=pre 제외)
        if self._im_post is None:
            self._init_artists(s)
        wl = self._wl
        self.lbl_sub.config(text=f"{s.label()}    slice {self.slice}/{s.n_slices-1}"
                                 f"    frame {self.frame}/{s.n_frames-1}")
        pre_s = s.volumes[0].data[self.slice]
        cur_s = s.volumes[self.frame].data[self.slice]
        self._im_pre.set_data(pre_s); self._im_pre.set_cmap("gray"); self._im_pre.set_clim(wl[0], wl[1])
        if self.sub:
            # 차감뷰: post 패널 = ΔS(이 시점-pre), 발산 컬러맵(빨강=증가/파랑=감소)
            self._im_post.set_data(cur_s - pre_s); self._im_post.set_cmap("seismic")
            self._im_post.set_clim(-self._sub_M, self._sub_M)
            self._heat_im.set_visible(False)
        else:
            self._im_post.set_data(cur_s); self._im_post.set_cmap("gray")
            self._im_post.set_clim(wl[0], wl[1])
            if self.heat:
                # 현재 frame 기준 증강맵: (이 시점 - pre)/pre %  (색 포화는 clim이 처리)
                E = np.where(pre_s > self._pre_thr,
                             (cur_s - pre_s) / (pre_s + 1e-6) * 100.0, np.nan)
                self._heat_im.set_data(E)
                self._heat_im.set_clim(self.hm_vmin, self.hm_vmax)
                self._heat_im.set_visible(True)
            else:
                self._heat_im.set_visible(False)
        self._mid_pre.set_xdata([self.mid, self.mid]); self._mid_post.set_xdata([self.mid, self.mid])
        tag = "pre-Gd" if self.frame == 0 else f"post-Gd {self.frame}"
        if self.sub:
            extra = "  ΔS(이 시점-pre)"
        elif self.heat:
            extra = "  +heatmap(이 시점-pre)"
        else:
            extra = ""
        self.ax_pre.set_title("pre-Gd", fontsize=10)
        self.ax_img.set_title(f"{tag}  (frame {self.frame}/{s.n_frames-1}){extra}", fontsize=10)
        for art in self._roi_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._roi_artists = []
        self._overlay_roi(self.ax_pre); self._overlay_roi(self.ax_img)
        self.canvas_img.draw_idle()

    def _draw_one(self, ax, roi, color, mirror=False):
        """ROI 외곽선 1개를 ax에. 추적 리스트에 넣어 다음 draw에서 제거."""
        if not (roi and roi["slice"] == self.slice):
            return
        if roi["kind"] == "circle":
            cx = 2 * self.mid - roi["cx"] if mirror else roi["cx"]
            c = self._Circle((cx, roi["cy"]), roi["r"], fill=False, color=color, lw=2)
            ax.add_patch(c); self._roi_artists.append(c)
        else:
            v = np.asarray(roi["verts"], float).copy()
            if mirror:
                v[:, 0] = 2 * self.mid - v[:, 0]
            if len(v) >= 2:
                vc = np.vstack([v, v[0]])
                l, = ax.plot(vc[:, 0], vc[:, 1], "-", color=color, lw=2)
                self._roi_artists.append(l)

    def _overlay_roi(self, ax):
        """누출부=빨강, 대조=시안(미러는 roi 반전, 수동은 roi2)."""
        self._draw_one(ax, self.roi, "red", mirror=False)
        if self.contra_mode == "manual":
            self._draw_one(ax, self.roi2, "cyan", mirror=False)
        else:
            self._draw_one(ax, self.roi, "cyan", mirror=True)

    # ---------- 토글 ----------
    def _toggle_heat(self):
        self.heat = bool(self.var_heat.get())
        if self.heat and self.sub:           # heat↔sub 상호배타
            self.sub = False; self.var_sub.set(False)
        self._draw_image()

    def _toggle_sub(self):
        self.sub = bool(self.var_sub.get())
        if self.sub and self.heat:
            self.heat = False; self.var_heat.set(False)
        self._draw_image()

    def _apply_hmscale(self):
        """히트맵 컬러 스케일 vmin~vmax 적용. 빈값/역전이면 무시."""
        try:
            vmn = float(self.var_vmin.get()); vmx = float(self.var_vmax.get())
        except ValueError:
            return
        if vmx <= vmn:
            self.lbl_sub.config(text="scale 오류: vmax > vmin 이어야 함"); return
        self.hm_vmin, self.hm_vmax = vmn, vmx
        if self.heat:
            self._draw_image()

    def _set_mode(self):
        self.mode = self.var_mode.get(); self._clear()

    def _set_contra(self):
        self.contra_mode = self.var_contra.get()
        if self.contra_mode == "mirror":        # 미러로 돌아가면 수동 대조·타겟 리셋
            self.roi2 = None
            self.target = "roi"; self.var_target.set("roi")
        self._draw_image(); self._update_curve()

    def _set_target(self):
        if self.var_contra.get() != "manual":
            self.var_target.set("roi")
            self.lbl_sub.config(text="'대조: 수동'으로 먼저 전환하세요"); return
        self.target = self.var_target.get()

    def _clear(self):
        self.roi = self.roi2 = None; self._poly = []
        self._draw_image(); self._update_curve()

    # ---------- ROI 그리기 (blitting) ----------
    def _busy(self):
        return False

    def _drawing_roi2(self):
        return self.contra_mode == "manual" and self.target == "roi2"

    def _on_press(self, e):
        if e.inaxes != self.ax_img or not self.slabs or self._busy():
            return
        self._r2 = self._drawing_roi2()
        color = "cyan" if self._r2 else "red"
        if self._r2:
            self.roi2 = None
        else:
            self.roi = None
        self._draw_image()
        canvas = self.canvas_img
        canvas.draw()                                  # 배경 픽셀 확정(드래그당 1회)
        self._bg = canvas.copy_from_bbox(self.ax_img.bbox)
        if self.mode == "circle":
            self._press = (e.xdata, e.ydata)
            self._drag_artist = self._Circle((e.xdata, e.ydata), 1.0, fill=False,
                                             color=color, lw=2, animated=True)
            self.ax_img.add_patch(self._drag_artist)
        else:
            self._press = True
            self._poly = [(e.xdata, e.ydata)]
            (self._drag_artist,) = self.ax_img.plot([e.xdata], [e.ydata], "-",
                                                    color=color, lw=2, animated=True)
        self._cur = None

    def _on_motion(self, e):
        if self._press is None or e.inaxes != self.ax_img or e.xdata is None:
            return
        if self.mode == "circle":
            cx, cy = self._press
            r = max(1.0, float(np.hypot(e.xdata - cx, e.ydata - cy)))
            self._drag_artist.set_center((cx, cy)); self._drag_artist.set_radius(r)
            self._cur = {"kind": "circle", "slice": self.slice, "cx": cx, "cy": cy, "r": r}
        else:
            self._poly.append((e.xdata, e.ydata))
            v = np.asarray(self._poly)
            self._drag_artist.set_data(v[:, 0], v[:, 1])
            self._cur = {"kind": "free", "slice": self.slice, "verts": list(self._poly)}
        canvas = self.canvas_img
        if self._bg is not None:                       # blit: 배경 복원 + 외곽선만
            canvas.restore_region(self._bg)
            self.ax_img.draw_artist(self._drag_artist)
            canvas.blit(self.ax_img.bbox)
        else:
            canvas.draw_idle()

    def _on_release(self, e):
        if self._press is None:
            return
        self._press = None; self._bg = None
        if self._drag_artist is not None:
            try:
                self._drag_artist.remove()
            except Exception:
                pass
            self._drag_artist = None
        if self._r2:
            self.roi2 = self._cur
        else:
            self.roi = self._cur
        self._cur = None
        active = self.roi2 if self._r2 else self.roi
        if self.mode == "free" and active and len(active.get("verts", [])) < 3:
            if self._r2:
                self.roi2 = None
            else:
                self.roi = None
        self._draw_image()
        if self.roi:
            self._update_curve()

    # ---------- 마스크/곡선 ----------
    def _one_mask(self, roi, shape, mirror=False):
        """ROI dict → bool 마스크. mirror=True면 정중선 기준 x 반전."""
        if not roi:
            return None
        sl = roi["slice"]
        if roi["kind"] == "circle":
            cx = 2 * self.mid - roi["cx"] if mirror else roi["cx"]
            return _disk_mask(shape, sl, cx, roi["cy"], roi["r"])
        v = np.asarray(roi["verts"], float).copy()
        if mirror:
            v[:, 0] = 2 * self.mid - v[:, 0]
        return _poly_mask(shape, sl, v)

    def _roi_masks(self):
        """(누출부 마스크, 대조 마스크). 대조=미러 자동 or 수동 roi2."""
        if not self.roi:
            return None, None
        shape = self.slabs[self.si].pre.data.shape
        rm = self._one_mask(self.roi, shape, mirror=False)
        if self.contra_mode == "manual":
            cm = self._one_mask(self.roi2, shape, mirror=False)
        else:
            cm = self._one_mask(self.roi, shape, mirror=True)
        return rm, cm

    def _curve(self, mask):
        s = self.slabs[self.si]
        sig = np.array([v.data[mask].mean() for v in s.volumes], float)
        base = sig[0] if sig[0] != 0 else 1.0
        return (sig - base) / base * 100.0

    def _update_curve(self):
        self.ax_cur.clear(); self.ax_cur.grid(alpha=0.3)
        self.ax_cur.set_xlabel("frame"); self.ax_cur.set_ylabel("% enhancement")
        if not self.slabs:
            self.canvas_cur.draw_idle(); return
        s = self.slabs[self.si]
        roi, mir = self._roi_masks()
        xs = range(1, s.n_frames)                # post1~ (pre 제거), post1=0% 기준 재정규화
        if roi is None or roi.sum() == 0:
            self.ax_cur.set_title("누출부 ROI를 그리세요"); self.canvas_cur.draw_idle(); return
        self.ax_cur.set_xlabel("frame (post1 기준)")
        cR = self._curve(roi); cRn = cR - cR[1]
        self.ax_cur.plot(xs, cRn[1:], "o-", color="red", label="누출부 ROI")
        if mir is not None and mir.sum() > 0:
            cM = self._curve(mir); cMn = cM - cM[1]; netn = cRn - cMn
            lbl = "대조(수동)" if self.contra_mode == "manual" else "대조(미러)"
            self.ax_cur.plot(xs, cMn[1:], "s-", color="cyan", label=lbl)
            self.ax_cur.plot(xs, netn[1:], "^--", color="green", label="net leakage")
            pk = float(netn[1:].max())
            verdict = "POSITIVE" if pk >= 15 else "NEG/equivocal"
            mm3 = int(roi.sum()) * s.voxel_mm3
            self.ax_cur.set_title(f"peak ROI {cRn[1:].max():.0f}%  net {pk:+.0f}%p "
                                  f"[{verdict}]  ROI {mm3:.2f}mm³")
            self._last = dict(cR=cRn, cM=cMn, net=netn, mm3=mm3, verdict=verdict,
                              roi=roi, mir=mir)
        else:
            self.ax_cur.set_title("누출부만 — 대조 ROI도 그리세요 (그릴곳: 대조)")
            self._last = None
        self.ax_cur.axhline(0, color="k", lw=0.5); self.ax_cur.legend(fontsize=8)
        self.ax_cur.set_xticks(list(xs))         # frame 정수 눈금만(0.5 간격 방지)
        self.canvas_cur.draw_idle()

    # ---------- 저장 ----------
    def _save_png(self):
        if not self.slabs:
            return
        out = self._ask_dir("PNG 저장 폴더 (pre 1 + post 시점별 개별 + heatmap 일렬)")
        if not out:
            return
        s = self.slabs[self.si]; z = self.slice; tag = s.label().replace(" ", "_")
        saved = save_frame_slices(s, z, out, wl=self._wl)
        hp = heatmap_frames_row(s, z, os.path.join(out, f"{tag}_slice{z}_heatmap_row.png"),
                                vmin=self.hm_vmin, vmax=self.hm_vmax)   # GUI scale 그대로
        sp = subtraction_frames_row(s, z, os.path.join(out, f"{tag}_slice{z}_subtraction_row.png"))
        print(f"저장: {out}")
        print(f"  개별 {len(saved)}장 = pre 1 + post {s.n_frames-1} (slice {z}, 고정윈도우)")
        print(f"  heatmap 일렬 (scale {self.hm_vmin:g}~{self.hm_vmax:g}): {os.path.basename(hp)}")
        print(f"  차감 일렬: {os.path.basename(sp)}")
        self.lbl_sub.config(text=f"저장: 개별 {len(saved)}장 + heatmap(scale {self.hm_vmin:g}~{self.hm_vmax:g}) + 차감 → {out}")

    def _save_all(self):
        """전체 슬랩을 한 번에 저장(각 슬랩 중앙 슬라이스). 슬랩 일일이 안 넘겨도 됨."""
        if not self.slabs:
            return
        out = self._ask_dir("전체 슬랩 저장 (각 슬랩 중앙 슬라이스, heatmap=현재 scale)")
        if not out:
            return
        cnt = 0
        for s in self.slabs:
            z = s.n_slices // 2
            wl = np.percentile(s.volumes[-1].data, [1, 99])
            tag = s.label().replace(" ", "_")
            save_frame_slices(s, z, out, wl=wl)
            heatmap_frames_row(s, z, os.path.join(out, f"{tag}_slice{z}_heatmap_row.png"),
                               vmin=self.hm_vmin, vmax=self.hm_vmax)
            subtraction_frames_row(s, z, os.path.join(out, f"{tag}_slice{z}_subtraction_row.png"))
            cnt += 1
            print(f"  저장 {s.label()} (slice {z})")
        print(f"전체 저장 완료: {cnt}슬랩 → {out}")
        self.lbl_sub.config(text=f"전체 {cnt}슬랩 저장(중앙 슬라이스, heatmap scale "
                                 f"{self.hm_vmin:g}~{self.hm_vmax:g}) → {out}")

    def _save_curve(self):
        if not self.slabs or not self.roi:
            self.lbl_sub.config(text="ROI 먼저 그리세요"); return
        if not self._last:
            self.lbl_sub.config(text="대조 ROI까지 그려야 곡선/CSV 저장 ('그릴곳: 대조')"); return
        out = self._ask_dir("그래프/CSV 저장 폴더")
        if not out:
            return
        s = self.slabs[self.si]; d = self._last
        base = os.path.join(out, f"{s.label().replace(' ', '_')}_BBBO")
        fig, ax = plt.subplots(figsize=(7, 5)); xs = range(1, s.n_frames)  # post1 기준 재정규화
        ax.plot(xs, d["cR"][1:], "o-", color="red", label="ROI")
        ax.plot(xs, d["cM"][1:], "s-", color="cyan", label="contralateral")
        ax.plot(xs, d["net"][1:], "^--", color="green", label="net leakage")
        ax.axhline(0, color="k", lw=0.5); ax.grid(alpha=0.3); ax.legend()
        ax.set_xlabel("frame (post1 기준)"); ax.set_ylabel("% enhancement")
        ax.set_xticks(list(xs))                   # frame 정수 눈금만
        ax.set_title(f"{s.label()}  net {d['net'][1:].max():+.0f}%p [{d['verdict']}]")
        fig.savefig(base + "_curve.png", dpi=130, bbox_inches="tight"); plt.close(fig)
        with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["frame", "ROI_enh%(post1)", "contra%(post1)", "net%(post1)"])
            for i in xs:
                w.writerow([i, round(d["cR"][i], 1), round(d["cM"][i], 1), round(d["net"][i], 1)])
        np.save(base + "_roi_mask.npy", d["roi"])
        np.save(base + "_contra_mask.npy", d["mir"])
        print("저장:", base + "_curve.png /", base + ".csv / *_mask.npy")
        self.lbl_sub.config(text=f"저장: 곡선+CSV+마스크 → {out}")


def run_gui(path=None, register=False):
    DCEApp(path, register=register).root.mainloop()


# ============================================================ 진입점
def main():
    argv = sys.argv[1:]
    do_reg = any(a in ("--register", "--reg") for a in argv)
    argv = [a for a in argv if a not in ("--register", "--reg")]
    if "--gui" in argv:
        rest = [a for a in argv if a != "--gui"]
        run_gui(rest[0] if rest else None, register=do_reg)
    elif len([a for a in argv if not a.startswith("-")]) >= 2:
        pos = [a for a in argv if not a.startswith("-")]
        run_batch(pos[0], pos[1], register=do_reg)
    else:
        run_gui()


if __name__ == "__main__":
    main()
