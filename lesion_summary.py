# -*- coding: utf-8 -*-
"""검출된 후보를 **실제 병변 단위로 통합**하고 인공물을 걸러낸다.

왜 필요한가
-----------
`local_focus`(국소 대비)와 `lateral_focus`(좌우 거울상)가 GRE·FSE × 축상·관상 =
네 번의 독립 획득에서 각각 후보를 낸다. 같은 병변이 여러 번 잡히고, 동시에 인공물도
같이 올라온다. 그림으로 하나씩 확인해 보니 가짜가 두 종류로 뚜렷했다.

1. **뇌 경계 테두리**. 증강 맵에서 뇌 윤곽을 따라 도는 밝은 띠가 생기고 후보가 그
   띠의 한 토막으로 잡힌다(폴더 1 gre coronal #1이 전형). PRE·POST 영상에는 그 자리에
   국소 증강이 보이지 않는다. 정합 잔차와 경막·뇌척수액 증강이 겹친 것이다.
   → 모양이 길쭉하고(경계를 따라가므로) 깊이가 얕고 깊이 편차가 작다.
2. **정맥동**. 정중선의 상시상정맥동·정맥동합류가 Gd로 크게 밝아진다(폴더 4 fse
   coronal #1이 전형). → 정중선에 붙어 있다.

그래서 여기서는 세 가지를 더 재서 같이 낸다.

* **길쭉함**(in-plane 2차 모멘트의 sqrt(λmax/λmin)). 테두리 토막은 크고, 진짜 초점은
  1~2 수준이다.
* **정중선까지 거리**. 정맥동을 거르는 데 쓴다.
* **독립 획득 재현 수**. 같은 물리 좌표를 GRE와 FSE, 축상과 관상에서 각각 잡았는지.
  네 획득은 서로 다른 시퀀스·다른 처방이라 같은 정합 잔차를 공유하지 않는다.
  **2개 이상에서 재현되면 인공물로 설명하기 어렵다.**

병변 통합은 환자 좌표계에서 중심 간 거리 `CLUSTER_MM` 이내를 같은 병변으로 묶는다.

실행: python "_lesion_summary.py"          # 폴더 1, 3, 4 전부
      python "_lesion_summary.py" 3        # 폴더 3만
"""
import os
import sys
import csv
import glob
import numpy as np
import pydicom
from scipy import ndimage

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))
DATA_ROOT = os.path.join(os.path.dirname(BASE), "260915_BBBO_120G")
sys.path.insert(0, os.path.join(ROOT, "MRI 분석 코드"))
from mri_dce import classify_orientation

# _run_0915.py 와 같은 값이어야 한다
FOCUS_ERODE = 21
LOCAL_BG_SIGMA = 20
LOCAL_EXCESS_PP = 15
LOCAL_MIN_VOX = 40
SHELL_IN, SHELL_OUT = 4, 12
DARK_FRAC = 0.85            # 뇌 마스크 문턱 (_focus_depth.py 와 동일)

CLUSTER_MM = 2.0            # 이 거리 안이면 같은 병변
MAX_ELONG = 3.0             # 이보다 길쭉하면 테두리 토막으로 본다
MIN_MIDLINE_MM = 0.8        # 정중선에서 이보다 가까우면 정맥동 의심
MAX_SHELL_PCT = 25.0        # 주변 고리가 이보다 크면 뇌가 아니라 근육

# pp=10 처럼 주면 국소 초과 문턱을 바꿔 감도 분석을 할 수 있다.
_args = [a for a in sys.argv[1:] if not a.startswith("pp=")]
for _a in sys.argv[1:]:
    if _a.startswith("pp="):
        LOCAL_EXCESS_PP = float(_a.split("=")[1])
SUFFIX = "" if LOCAL_EXCESS_PP == 15 else f"_pp{int(LOCAL_EXCESS_PP)}"
RATS = _args or ["1", "3", "4"]


def disk(rr):
    y, x = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    return (y ** 2 + x ** 2 <= rr * rr)[None, :, :]


def brain_mask(pre, tissue):
    med = float(np.median(pre[tissue]))
    bright = ndimage.binary_opening(tissue & (pre >= DARK_FRAC * med),
                                    np.ones((1, 3, 3)))
    lab, n = ndimage.label(bright)
    if n == 0:
        return None
    sizes = ndimage.sum(bright, lab, range(1, n + 1))
    br = lab == (int(np.argmax(sizes)) + 1)
    return np.stack([ndimage.binary_fill_holes(br[z]) for z in range(br.shape[0])])


def geom(rat_dir, kind, ori):
    """(첫 슬라이스 IPP, r, c, n, 화소크기, 슬라이스 간격). PRE 기준."""
    for folder in sorted(os.listdir(rat_dir)):
        d = os.path.join(rat_dir, folder)
        if not os.path.isdir(d):
            continue
        up = folder.upper()
        if kind.upper() not in up or "_PRE" not in up:
            continue
        fs = sorted(glob.glob(os.path.join(d, "*.dcm")))
        ds = [pydicom.dcmread(f, stop_before_pixels=True) for f in fs]
        iop = np.array([float(x) for x in ds[0].ImageOrientationPatient])
        if classify_orientation(iop) != ori:
            continue
        r, c = iop[:3], iop[3:]
        n = np.cross(r, c)
        ds.sort(key=lambda x: float(np.dot(
            np.array([float(v) for v in x.ImagePositionPatient]), n)))
        ipp = np.array([float(v) for v in ds[0].ImagePositionPatient])
        ps = float(ds[0].PixelSpacing[0])
        sz = float(getattr(ds[0], "SpacingBetweenSlices", None) or ds[0].SliceThickness)
        return ipp, r, c, n, ps, sz
    return None


all_rows, cluster_rows = [], []
for rat in RATS:
    out_dir = os.path.join(BASE, f"RAT {rat}")
    npz = os.path.join(out_dir, "corrected_volumes.npz")
    if not os.path.exists(npz):
        continue
    d = np.load(npz)
    data_dir = os.path.join(DATA_ROOT, rat)
    print("=" * 78)
    print(f"폴더 {rat}")
    cands = []
    for kind in ("gre", "fse"):
        for ori in ("axial", "coronal"):
            k = f"{kind}_{ori}_PRE"
            if k not in d.files:
                continue
            tps = ["PRE"] + sorted(
                [x.split("_")[-1] for x in d.files
                 if x.startswith(f"{kind}_{ori}_POST")],
                key=lambda x: int(x.replace("POST", "") or 1))
            vols = [d[f"{kind}_{ori}_{t}"].astype(float) for t in tps]
            pre, last = vols[0], vols[-1]
            m = d[f"{kind}_{ori}_analysis_mask"]
            sp = d[f"{kind}_{ori}_spacing"]
            g = geom(data_dir, kind, ori)
            br = brain_mask(pre, m)
            if g is None or br is None:
                print(f"  [경고] {kind} {ori}: 기하 또는 뇌 마스크 실패")
                continue
            ipp, rv, cv, nv, ps, sz = g
            dist = ndimage.distance_transform_edt(
                br, sampling=(float(sp[2]), float(sp[1]), float(sp[0])))
            enh = np.where(m, (last - pre) / (pre + 1e-6) * 100.0, 0.0)
            num = ndimage.gaussian_filter(enh, (0, LOCAL_BG_SIGMA, LOCAL_BG_SIGMA))
            den = ndimage.gaussian_filter(m.astype(float), (0, LOCAL_BG_SIGMA, LOCAL_BG_SIGMA))
            exc = ndimage.gaussian_filter(np.where(m, enh - num / np.maximum(den, 1e-6), 0.0),
                                          (0, 1.5, 1.5))
            inner = ndimage.binary_erosion(m, np.ones((1, FOCUS_ERODE, FOCUS_ERODE)))
            cand = ndimage.binary_opening(inner & (exc > LOCAL_EXCESS_PP), np.ones((1, 3, 3)))
            lab, n = ndimage.label(cand)
            if n == 0:
                continue
            sizes = ndimage.sum(cand, lab, range(1, n + 1))
            vox = abs(sp[0] * sp[1] * sp[2])
            din, dout = disk(SHELL_IN), disk(SHELL_OUT)
            # 정중선 열: 뇌 마스크의 열 중심
            midcol = float(np.argwhere(br)[:, 2].mean())
            for kk in np.argsort(sizes)[::-1][:6]:
                if sizes[kk] < LOCAL_MIN_VOX:
                    break
                sel = lab == (kk + 1)
                shell = (ndimage.binary_dilation(sel, dout)
                         & ~ndimage.binary_dilation(sel, din) & inner)
                if shell.sum() < 50:
                    continue
                pts = np.argwhere(sel)
                zc, rc, cc = pts.mean(0)
                sig = [float(v[sel].mean()) for v in vols]
                sgs = [float(v[shell].mean()) for v in vols]
                e = (sig[-1] - sig[0]) / sig[0] * 100.0
                es = (sgs[-1] - sgs[0]) / sgs[0] * 100.0
                # 면내 2차 모멘트로 길쭉함
                xy = pts[:, 1:].astype(float) * np.array([sp[1], sp[0]])
                cov = np.cov(xy.T) if len(xy) > 2 else np.eye(2)
                ev = np.sort(np.linalg.eigvalsh(cov))
                elong = float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))
                dsel = dist[sel]
                inside = bool(br[int(round(zc)), int(round(rc)), int(round(cc))])
                zs = sorted({int(z_) for z_ in pts[:, 0]})
                edge = (min(zs) == 0 or max(zs) == pre.shape[0] - 1)
                P = ipp + cc * ps * rv + rc * ps * cv + zc * sz * nv
                cands.append(dict(
                    kind=kind, ori=ori, vol=float(sizes[kk] * vox),
                    net=e - es, shell=es, elong=elong,
                    depth=float(dsel.mean()), depth_max=float(dsel.max()),
                    inside=inside, edge=edge, nslice=len(zs), slices=zs,
                    mid_mm=abs(cc - midcol) * float(sp[0]), P=P,
                    z=zc, row=rc, col=cc))

    # --- 후보 거르기
    def keep(x):
        # 길쭉함과 정중선 거리는 **탈락 조건이 아니라 표시**다. 처음에는 탈락시켰더니
        # 육안으로 확인된 진짜 병소(폴더 3 gre axial 21.75mm3, 정중선에서 0.30mm)까지
        # 걸러졌다. 하드 조건은 물리적으로 말이 안 되는 것만 남긴다.
        return (x["inside"] and not x["edge"] and x["net"] >= 15.0
                and 0 <= x["shell"] < MAX_SHELL_PCT)

    good = [x for x in cands if keep(x)]
    for x in cands:
        why = []
        if not x["inside"]:
            why.append("뇌밖")
        if x["edge"]:
            why.append("가장자리")
        if x["net"] < 15:
            why.append("누출<15")
        if not (0 <= x["shell"] < MAX_SHELL_PCT):
            why.append("고리근육")
        flag = []
        if x["elong"] > MAX_ELONG:
            flag.append(f"길쭉{x['elong']:.1f}(테두리 의심)")
        if x["mid_mm"] < MIN_MIDLINE_MM:
            flag.append("정중선 인접(정맥동 의심)")
        all_rows.append([rat, x["kind"], x["ori"], round(x["vol"], 2), round(x["net"], 1),
                         round(x["shell"], 1), round(x["elong"], 2), round(x["depth"], 2),
                         round(x["mid_mm"], 2), x["nslice"],
                         " ".join(str(v) for v in x["slices"]),
                         ("통과" if not why else ",".join(why))
                         + ((" | " + ",".join(flag)) if flag else "")])
    print(f"  후보 {len(cands)}개 중 통과 {len(good)}개")

    # --- 물리 좌표로 묶기
    used = [False] * len(good)
    clusters = []
    for i, a in enumerate(good):
        if used[i]:
            continue
        grp = [i]
        used[i] = True
        for j in range(i + 1, len(good)):
            if used[j]:
                continue
            if np.linalg.norm(a["P"] - good[j]["P"]) <= CLUSTER_MM:
                grp.append(j)
                used[j] = True
        clusters.append(grp)
    clusters.sort(key=lambda g: -max(good[i]["vol"] for i in g))

    for ci, g in enumerate(clusters, 1):
        mem = [good[i] for i in g]
        acq = sorted({f"{x['kind']}-{x['ori'][:3]}" for x in mem})
        el = float(np.mean([x["elong"] for x in mem]))
        md = float(np.mean([x["mid_mm"] for x in mem]))
        note = []
        if el > MAX_ELONG:
            note.append("테두리 의심")
        if md < MIN_MIDLINE_MM:
            note.append("정맥동 의심")
        conf = ("높음" if len(acq) >= 2 else "낮음") + (" (" + ",".join(note) + ")" if note else "")
        vol = max(x["vol"] for x in mem)
        dep = float(np.mean([x["depth"] for x in mem]))
        net = float(np.mean([x["net"] for x in mem]))
        P = np.mean([x["P"] for x in mem], axis=0)
        print(f"  병변 {ci}: 최대 부피 {vol:6.2f} mm3  뇌 표면 아래 {dep:.2f} mm  "
              f"순 누출 평균 {net:+.1f} %p  획득 {len(acq)}개 {acq}  신뢰 {conf}")
        cluster_rows.append([rat, ci, round(vol, 2), round(dep, 2), round(net, 1),
                             len(acq), " ".join(acq), conf,
                             round(float(np.mean([x["elong"] for x in mem])), 2),
                             round(float(np.mean([x["mid_mm"] for x in mem])), 2),
                             " ".join(f"{v:.2f}" for v in P)])

with open(os.path.join(BASE, f"lesion_candidates_all{SUFFIX}.csv"), "w",
          newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["rat_folder", "kind", "orientation", "volume_mm3", "net_leakage_pp",
                "shell_enh_pct", "elongation", "depth_mm", "dist_from_midline_mm",
                "n_slices", "slices", "verdict"])
    w.writerows(all_rows)
with open(os.path.join(BASE, f"lesion_summary{SUFFIX}.csv"), "w",
          newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["rat_folder", "lesion", "max_volume_mm3", "depth_below_brain_surface_mm",
                "mean_net_leakage_pp", "n_independent_acquisitions", "acquisitions",
                "confidence", "elongation", "dist_from_midline_mm", "patient_xyz_mm"])
    w.writerows(cluster_rows)
print("\n저장: lesion_candidates_all.csv, lesion_summary.csv")
