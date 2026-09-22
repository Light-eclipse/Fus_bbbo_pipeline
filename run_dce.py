# -*- coding: utf-8 -*-
"""260922 BBBO 200G — 시리즈별 강도 스케일 보정 분석 (폴더 1, 4). GRE·FSE 부분.

절차는 26-09-15판(`_run_0915.py`)과 같다. `참값 = 화소값 / 태그(00E1,1001)`로 되돌린 뒤
GRE(gems)와 FSE(fsems)를 각각 독립 시계열로 처리한다. 좌우 대칭을 가정하지 않는
국소 대비 검출(`local_focus`)도 그대로 돈다.

이 날짜의 특징
--------------
1. **파일명과 폴더 번호가 엇갈린다.** 폴더 1의 파일은 전부 `NO2`, 폴더 4는 `NO1`이다.
   촬영은 폴더 1이 먼저(10:15), 폴더 4가 나중(11:45)이다. 개체는 **폴더 번호로만**
   구분한다. 두 폴더는 IPP·IOP와 촬영 시간대가 전부 달라 서로 다른 개체다.
2. PRE와 POST 사이에 **IVIM(SE-EPI, b 11개)**이 끼어 있다. PRE 후 약 34분 공백,
   IVIM 약 8~12분, 다시 약 10~13분 공백 뒤 GRE POST1이다. IVIM은 이 스크립트가 아니라
   `_ivim_0922.py`가 처리한다.
3. **MREPT(8에코 NIfTI)**와 `recon.nii`가 있다. 둘은 **파일 단위로 동일**하다(8에코 전부
   md5 일치). 위상 영상이 없으므로 EPT는 계산할 수 없다. T2는 `_t2_0922.py`가 처리한다.
4. PRE와 POST의 IPP가 폴더별로 소수점까지 같다.

실행
----
    python "_run_0922.py"          # 폴더 1, 4 전부
    python "_run_0922.py" 4        # 폴더 4만
    python "_run_0922.py" figs     # 저장된 npz로 그림·정량만 다시
"""
import os
import re
import sys
import csv
import glob
from dataclasses import replace
import numpy as np
import pydicom
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\user\Desktop\대학원\6. 원자력의학원"
CODE_DIR = os.path.join(ROOT, "MRI 분석 코드")
BASE = os.path.join(ROOT, "26-09-22")
DATA_ROOT = os.path.join(BASE, "260922_BBBO_200G")
OUT_ROOT = os.path.join(BASE, "2. 분석 결과")

SCALE_TAG = (0x00E1, 0x1001)
AIR_PCT = 10           # 배경 마스크 백분위수
HEAT_VMAX = 100        # 슬라이스별 히트맵 컬러 상한 %
HEAT_VMAX_BRAIN = 40   # 뇌실질 감도용 몽타주 상한 %
FOCUS_ASYM_PP = 20     # 편측 후보 문턱 (%p)
FOCUS_MIN_VOX = 40     # 최소 화소수
FOCUS_ERODE = 21       # 두피/근육 제외용 침식 커널(화소)
MIRROR_DS = 2          # 대칭축 탐색 다운샘플 배수
MIRROR_MAX_DEG = 10    # 탐색할 최대 기울기 (도)

KINDS = ("gre", "fse")
KIND_LABEL = {"gre": "GRE DCE (gems)", "fse": "FSE T1 (fsems)"}

# 파일명(폴더1=NO2, 폴더4=NO1)과 폴더 번호가 엇갈려 폴더 번호를 식별자로 쓴다.
SETS = {r: dict(data=os.path.join(DATA_ROOT, r),
                out=os.path.join(OUT_ROOT, f"RAT {r}"),
                label=f"0922_F{r}") for r in ("1", "4")}

sys.path.insert(0, CODE_DIR)
from mri_dce import (_load_volumes, Slab, register_slab, enhancement_curve,
                     classify_orientation, plot_curves, save_frame_slices,
                     heatmap_frames_row, subtraction_frames_row,
                     montage_anatomy, montage_heatmap)


# ------------------------------------------------------------ 시리즈 분류
def classify(folder):
    """폴더명 → (계열, 시점). 계열 'gre' | 'fse', 시점 PRE=0 / POSTn=n.

    `_tp_rank`를 쓰지 않는다. GRE와 FSE를 구분하지 못하고, 둘을 한 슬랩에 섞으면
    프로토콜이 다른 영상이 같은 시계열로 들어간다.
    """
    f = folder.upper()
    m = re.search(r"(GRE|FSE)_(?:AX|COR)_(PRE|POST)(\d*)_", f)
    if not m:
        return "other", None
    kind = "gre" if m.group(1) == "GRE" else "fse"
    if m.group(2) == "PRE":
        return kind, 0
    return kind, int(m.group(3)) if m.group(3) else 1


def series_meta(rat_dir):
    """시리즈 UID → 최소투영 슬라이스 IPP / 스케일 인자 / 폴더명."""
    ipp_out, scale_out, name_out = {}, {}, {}
    for root, _, fs in os.walk(rat_dir):
        for f in fs:
            fp = os.path.join(root, f)
            try:
                ds = pydicom.dcmread(fp, force=True, stop_before_pixels=True)
            except Exception:
                continue
            uid = getattr(ds, "SeriesInstanceUID", None)
            ipp = getattr(ds, "ImagePositionPatient", None)
            iop = getattr(ds, "ImageOrientationPatient", None)
            if uid is None or ipp is None or iop is None:
                continue
            r, c = np.array(iop[:3], float), np.array(iop[3:], float)
            n = np.cross(r, c)
            ipp = np.array([float(x) for x in ipp])
            p = float(np.dot(ipp, n))
            if uid not in ipp_out or p < ipp_out[uid][0]:
                ipp_out[uid] = (p, ipp)
            if uid not in scale_out:
                try:
                    scale_out[uid] = float(ds[SCALE_TAG].value)
                except Exception:
                    scale_out[uid] = 1.0
                    print(f"  [경고] {os.path.basename(root)}: 스케일 태그 없음 -> 1.0",
                          flush=True)
                name_out[uid] = os.path.basename(root)
    return {uid: ipp for uid, (p, ipp) in ipp_out.items()}, scale_out, name_out


def series_params(rat_dir, folder):
    fs = sorted(glob.glob(os.path.join(rat_dir, folder, "*.dcm")))
    ds = pydicom.dcmread(fs[0], stop_before_pixels=True)
    return dict(n=len(fs), seq=str(ds.get("SequenceName", "?")),
                tr=float(ds.RepetitionTime), te=float(ds.EchoTime),
                fa=float(ds.FlipAngle), nex=float(ds.NumberOfAverages),
                etl=ds.get("EchoTrainLength", "-"),
                mtx=int(ds.Rows), px=float(ds.PixelSpacing[0]),
                t=str(ds.get("ContentTime", "?")))


def resample_to(pre_v, mov_v, ippmin):
    """mov 볼륨을 pre 격자로 평행이동 리샘플(IOP·spacing 동일 전제).

    0.01 미만 성분은 0으로 스냅한다. ndimage.shift는 dk=-4e-5 같은 미세 음수에도
    인덱스를 경계 밖으로 판정해 가장자리 슬라이스 전체를 cval(0)로 채운다.
    """
    r = np.array(pre_v.iop[:3], float)
    c = np.array(pre_v.iop[3:], float)
    n = np.cross(r, c)
    d = ippmin[pre_v.series_uid] - ippmin[mov_v.series_uid]
    sx, sy, sz = pre_v.spacing
    dk = float(np.dot(d, n)) / sz
    dj = float(np.dot(d, c)) / sy
    di = float(np.dot(d, r)) / sx
    dk, dj, di = (0.0 if abs(x) < 0.01 else x for x in (dk, dj, di))
    if dk == 0.0 and dj == 0.0 and di == 0.0:
        return replace(mov_v, origin=pre_v.origin), (0.0, 0.0, 0.0)
    shifted = ndimage.shift(mov_v.data.astype(np.float32), (-dk, -dj, -di),
                            order=1, mode="constant", cval=0.0)
    return replace(mov_v, data=shifted, origin=pre_v.origin), (dk, dj, di)


def zoom_to(arr, ny, nx):
    """면내 격자만 맞춘다(화소 면적 정렬). 표시용."""
    if arr.shape[1] == ny and arr.shape[2] == nx:
        return arr
    return ndimage.zoom(arr, (1, ny / arr.shape[1], nx / arr.shape[2]),
                        order=1, grid_mode=True, mode="nearest")


def tp_name(v):
    return "PRE" if v.tp == 0 else f"POST{v.tp}"


def key_of(kind, ori, name=""):
    return f"{kind}_{ori}" + (f"_{name}" if name else "")


# ------------------------------------------------------------ 개체 1마리 처리
def run_one(rat, cfg):
    DATA_DIR, OUT_DIR, ANIMAL = cfg["data"], cfg["out"], cfg["label"]
    print(f"\n########## RAT {rat}  ({ANIMAL}) ##########", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    vols_raw = _load_volumes(DATA_DIR, True)
    ippmin, scale, sname = series_meta(DATA_DIR)
    vols = []
    for v in vols_raw:
        kind, tp = classify(sname[v.series_uid])
        if kind == "other":
            print(f"  [경고] 분류 실패: {sname[v.series_uid]} -> 제외", flush=True)
            continue
        vols.append((kind, replace(v, data=v.data / scale[v.series_uid], tp=tp)))

    # --- 스캔 프로토콜 표
    with open(os.path.join(OUT_DIR, "series_protocol.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["kind", "orientation", "timepoint", "series_folder", "content_time",
                    "sequence", "TR_ms", "TE_ms", "flip_deg", "NEX", "ETL", "matrix",
                    "pixel_mm", "n_slices", "scale_tag_00E1_1001"])
        for k, v in sorted(vols, key=lambda x: series_params(
                DATA_DIR, sname[x[1].series_uid])["t"]):
            p = series_params(DATA_DIR, sname[v.series_uid])
            w.writerow([k, v.orientation, tp_name(v), sname[v.series_uid], p["t"],
                        p["seq"], p["tr"], p["te"], p["fa"], p["nex"], p["etl"],
                        f"{p['mtx']}x{p['mtx']}", round(p["px"], 6), p["n"],
                        round(scale[v.series_uid], 4)])

    # --- 스케일 보정 검증 (계열·방향별로 프로토콜이 같아야 배경 잡음이 불변량)
    check_rows, check_plot = [], {}
    for kind in KINDS:
        for ori in sorted({v.orientation for k, v in vols if k == kind}):
            vs = sorted([v for k, v in vols if k == kind and v.orientation == ori],
                        key=lambda v: v.sort_key())
            if len({v.data.shape for v in vs}) > 1:
                print(f"  [경고] {kind} {ori}: 시점 간 shape 불일치 -> 배경 검증 생략",
                      flush=True)
                continue
            mean = np.mean([v.data for v in vs], axis=0)
            air = mean < np.percentile(mean, AIR_PCT)
            raw, cor = [], []
            for v in vs:
                vr = next(r for r in vols_raw if r.series_uid == v.series_uid)
                raw.append(float(np.median(vr.data[air])))
                cor.append(float(np.median(v.data[air])))
                check_rows.append([kind, ori, tp_name(v), sname[v.series_uid],
                                   round(scale[v.series_uid], 4),
                                   round(raw[-1], 1), round(cor[-1], 3)])
            check_plot[(kind, ori)] = ([tp_name(v) for v in vs], raw, cor)
            dev = max(abs(x / cor[0] * 100 - 100) for x in cor)
            devr = max(abs(x / raw[0] * 100 - 100) for x in raw)
            print(f"  {kind} {ori}: 배경 잡음 편차 보정 전 {devr:.1f}% -> 보정 후 {dev:.1f}%",
                  flush=True)

    with open(os.path.join(OUT_DIR, "scale_factors.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["kind", "orientation", "timepoint", "series_folder",
                    "scale_tag_00E1_1001", "bg_median_raw", "bg_median_corrected"])
        w.writerows(check_rows)

    if check_plot:
        ks = sorted(check_plot)
        fig, axes = plt.subplots(1, len(ks), figsize=(4.6 * len(ks), 4), squeeze=False)
        for ax, kk in zip(axes[0], ks):
            labs, raw, cor = check_plot[kk]
            x = np.arange(len(labs))
            ax.plot(x, np.array(raw) / raw[0] * 100, "o-", label="보정 전 (raw)")
            ax.plot(x, np.array(cor) / cor[0] * 100, "s-", label="보정 후 (/태그)")
            ax.axhline(100, color="k", lw=0.6, ls=":")
            ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
            ax.set_ylabel("배경 잡음 중앙값 (PRE=100%)")
            ax.set_title(f"{kk[0]} {kk[1]}", fontsize=10)
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle(f"{ANIMAL} 스케일 보정 검증: 불변이어야 할 공기 배경 잡음", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "scale_check_background.png"), dpi=130)
        plt.close(fig)

    # --- 계열·방향별 Slab 구성 + 정합
    report, SLABS, MASKS, ZS = [], {}, {}, {}
    for kind in KINDS:
        for ori in sorted({v.orientation for k, v in vols if k == kind}):
            vs = sorted([v for k, v in vols if k == kind and v.orientation == ori],
                        key=lambda v: v.sort_key())
            if not vs or vs[0].tp != 0 or len(vs) < 2:
                print(f"  [경고] {kind} {ori}: PRE 또는 POST 없음 -> 건너뜀", flush=True)
                continue
            pre_v, posts = vs[0], vs[1:]
            merged, dks = [pre_v], []
            for pv in posts:
                mv, (dk, dj, di) = resample_to(pre_v, pv, ippmin)
                merged.append(mv); dks.append(dk)
                if abs(dk) > 1e-3 or abs(dj) > 1e-3 or abs(di) > 1e-3:
                    print(f"  {kind} {ori} {tp_name(pv)}: PRE 격자로 리샘플 "
                          f"(면외 {dk:+.2f} slice, 면내 {dj:+.2f}/{di:+.2f} px)", flush=True)
            nz = pre_v.n_slices
            dks = [0.0 if abs(k) < 0.01 else k for k in dks]
            lo = max(0, int(np.ceil(max(0.0, *[-k for k in dks]))))
            hi = min(nz - 1, int(np.floor(min(float(nz - 1), *[nz - 1 - k for k in dks]))))
            slab = Slab(volumes=merged, orientation=ori, animal=f"{ANIMAL}_{kind}")
            slab = register_slab(slab, "inplane", True, verbose=True)
            pre = slab.pre.data
            tis = pre > float(np.mean(pre[pre > 0]))
            support = np.all(np.stack([v.data for v in slab.volumes]) > 0, axis=0)
            m = tis & support
            keep = [z for z in range(slab.n_slices)
                    if tis[z].sum() > 0 and m[z].sum() >= 0.5 * tis[z].sum()]
            drop = 100.0 * (1.0 - m.sum() / max(int(tis.sum()), 1))
            print(f"  {kind} {ori}: 시점 {[tp_name(v) for v in merged]}, "
                  f"조직 화소의 {drop:.1f}%가 0채움이라 제외, "
                  f"분석 슬라이스 {keep[0]}~{keep[-1]} ({len(keep)}/{slab.n_slices})",
                  flush=True)
            SLABS[(kind, ori)] = slab
            MASKS[(kind, ori)] = m
            ZS[(kind, ori)] = keep
            report.append([ANIMAL, kind, ori, nz, lo, hi,
                           round(max([abs(k) for k in dks] or [0.0]), 3),
                           " ".join(tp_name(v) for v in merged),
                           round(drop, 2), f"{keep[0]}~{keep[-1]}", len(keep)])

    with open(os.path.join(OUT_DIR, "valid_slice_ranges.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rat", "kind", "orientation", "n_slices", "shift_lo", "shift_hi",
                    "max_shift_slice", "timepoints", "dropped_tissue_vox_pct",
                    "analysis_slices", "n_analysis_slices"])
        w.writerows(report)

    # --- 보정·정합 끝난 볼륨 저장 (시점 간 절대 비교 가능)
    npz = {}
    for (kind, ori), s in SLABS.items():
        for v in s.volumes:
            npz[key_of(kind, ori, tp_name(v))] = v.data.astype(np.float32)
        npz[key_of(kind, ori, "analysis_mask")] = MASKS[(kind, ori)]
        npz[key_of(kind, ori, "spacing")] = np.array(s.spacing, float)
    np.savez_compressed(os.path.join(OUT_DIR, "corrected_volumes.npz"), **npz)

    # --- 증강 곡선 (계열별로 따로. 시퀀스가 다르므로 서로 비교하지 않는다)
    summary, curves_by_kind = [], {}
    with open(os.path.join(OUT_DIR, "enhancement_summary.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rat", "kind", "orientation", "n_frames", "roi_mm3",
                    "signal_by_frame", "enh_pct_by_frame", "peak_pct",
                    "time_to_peak_frame", "washin", "auc"])
        for (kind, ori), s in sorted(SLABS.items()):
            cv = enhancement_curve(s, MASKS[(kind, ori)])
            curves_by_kind.setdefault(kind, []).append(cv)
            row = [rat, kind, ori, cv["n_frames"], cv["roi_mm3"],
                   " ".join(f"{x:.1f}" for x in cv["signal"]),
                   " ".join(f"{x:+.1f}" for x in cv["enh_pct"]),
                   cv["peak_pct"], cv["time_to_peak_frame"],
                   cv["washin_pct_per_unit"], cv["auc_pct"]]
            w.writerow(row)
            summary.append(row)
            print(f"  [{kind} {ori}] enh% = {cv['enh_pct']}  peak={cv['peak_pct']}%",
                  flush=True)
    for kind, cvs in curves_by_kind.items():
        plot_curves(cvs, os.path.join(OUT_DIR, f"all_curves_{kind}.png"))

    # --- 전 슬라이스 추출 + 마지막 POST 차감
    stat_rows = []
    for (kind, ori), s in sorted(SLABS.items()):
        d_k = os.path.join(OUT_DIR, kind)
        tag = f"{ANIMAL}_{kind}_{ori}"
        os.makedirs(d_k, exist_ok=True)
        montage_anatomy(s, "pre", os.path.join(d_k, f"{tag}_preGd_anatomy.png"))
        montage_anatomy(s, "peak", os.path.join(d_k, f"{tag}_peakGd_anatomy.png"))
        montage_heatmap(s, os.path.join(d_k, f"{tag}_heatmap.png"))

        d_ind = os.path.join(d_k, ori, "individual")
        d_hm = os.path.join(d_k, ori, "heatmap_rows")
        d_sub = os.path.join(d_k, ori, "subtraction_rows")
        for d in (d_ind, d_hm, d_sub):
            os.makedirs(d, exist_ok=True)
        wl = np.percentile(s.volumes[-1].data, [1, 99])
        for z in range(s.n_slices):
            save_frame_slices(s, z, d_ind, wl=wl)
            heatmap_frames_row(s, z, os.path.join(d_hm, f"{tag}_slice{z:02d}_heatmap_row.png"))
            subtraction_frames_row(s, z, os.path.join(d_sub, f"{tag}_slice{z:02d}_subtraction_row.png"))

        # PRE vs 마지막 POST
        last = s.volumes[-1]
        d_p = os.path.join(d_k, f"pre_vs_{tp_name(last)}")
        d_heat = os.path.join(d_p, "heatmap")
        for d in (d_p, d_heat):
            os.makedirs(d, exist_ok=True)
        pre = s.volumes[0].data.astype(float)
        pl = last.data.astype(float)
        delta = pl - pre
        tis = MASKS[(kind, ori)]
        enh = np.where(tis, delta / (pre + 1e-6) * 100.0, np.nan)
        zs = ZS[(kind, ori)]
        wl = np.percentile(pl[zs], [1, 99])
        M = float(np.percentile(np.abs(delta[tis]), 99)) or 1.0
        vox_mm2 = abs(s.spacing[0] * s.spacing[1])
        for z in zs:
            mz = tis[z]
            fig, ax = plt.subplots(figsize=(5.2, 5.0))
            im = ax.imshow(delta[z], cmap="seismic", vmin=-M, vmax=M)
            ax.axis("off")
            ax.set_title(f"{tag}  slice {z}   dS = {tp_name(last)} - PRE", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="dS (빨강=증가)")
            fig.savefig(os.path.join(d_p, f"{tag}_slice{z:02d}_delta.png"),
                        dpi=140, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(5.2, 5.0))
            ax.imshow(pl[z], cmap="gray", vmin=wl[0], vmax=wl[1])
            im = ax.imshow(enh[z], cmap="jet", vmin=0, vmax=HEAT_VMAX, alpha=0.55)
            ax.axis("off")
            ax.set_title(f"{tag}  slice {z}   ({tp_name(last)}-PRE)/PRE", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="% 증강")
            fig.savefig(os.path.join(d_heat, f"{tag}_slice{z:02d}_heat.png"),
                        dpi=140, bbox_inches="tight")
            plt.close(fig)

            e = enh[z][mz]; dd = delta[z][mz]
            stat_rows.append([
                kind, ori, z, int(mz.sum()),
                round(float(np.mean(pre[z][mz])), 1), round(float(np.mean(pl[z][mz])), 1),
                round(float(np.mean(dd)), 2), round(float(np.percentile(dd, 95)), 2),
                round(float(np.mean(e)), 1), round(float(np.median(e)), 1),
                round(float(np.percentile(e, 95)), 1), round(float(np.max(e)), 1),
                int((e > 15).sum()), int((e > 25).sum()), int((e > 50).sum()),
                round(float((e > 25).sum()) * vox_mm2, 2)])

        n = len(zs); cols = min(7, n); rows_ = int(np.ceil(n / cols))
        for vmax, suffix in ((HEAT_VMAX, ""), (HEAT_VMAX_BRAIN, f"_vmax{HEAT_VMAX_BRAIN}")):
            fig, axes = plt.subplots(rows_, cols, figsize=(2.6 * cols, 2.8 * rows_),
                                     squeeze=False)
            axes = axes.ravel(); im = None
            for k2, z in enumerate(zs):
                axes[k2].imshow(pl[z], cmap="gray", vmin=wl[0], vmax=wl[1])
                im = axes[k2].imshow(enh[z], cmap="jet", vmin=0, vmax=vmax, alpha=0.55)
                axes[k2].set_title(f"slice {z}", fontsize=8)
            for a in axes:
                a.axis("off")
            fig.suptitle(f"{tag}  ({tp_name(last)}-PRE)/PRE  "
                         f"(분석 슬라이스 {zs[0]}~{zs[-1]}, 상한 {vmax}%)", fontsize=12)
            fig.subplots_adjust(right=0.9)
            fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.010, 0.7]), label="% 증강")
            fig.savefig(os.path.join(d_p, f"{tag}_montage_heat{suffix}.png"),
                        dpi=130, bbox_inches="tight")
            plt.close(fig)

        fig, axes = plt.subplots(rows_, cols, figsize=(2.6 * cols, 2.8 * rows_), squeeze=False)
        axes = axes.ravel(); im = None
        for k2, z in enumerate(zs):
            im = axes[k2].imshow(delta[z], cmap="seismic", vmin=-M, vmax=M)
            axes[k2].set_title(f"slice {z}", fontsize=8)
        for a in axes:
            a.axis("off")
        fig.suptitle(f"{tag}  dS = {tp_name(last)} - PRE (빨강=증가)", fontsize=12)
        fig.subplots_adjust(right=0.9)
        fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.010, 0.7]), label="dS")
        fig.savefig(os.path.join(d_p, f"{tag}_montage_delta.png"), dpi=130,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  {tag}: 전체 {s.n_slices}슬라이스 + 차감 저장  "
              f"조직 평균 PRE={pre[tis].mean():.1f} -> {pl[tis].mean():.1f} "
              f"({delta[tis].mean() / pre[tis].mean() * 100:+.1f}%)", flush=True)

    with open(os.path.join(OUT_DIR, "pre_vs_lastpost_slice_stats.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["kind", "orientation", "slice_0idx", "tissue_vox",
                    "pre_mean", "lastpost_mean", "dS_mean", "dS_p95",
                    "enh_mean_pct", "enh_median_pct", "enh_p95_pct", "enh_max_pct",
                    "vox_enh_gt15", "vox_enh_gt25", "vox_enh_gt50", "area_gt25_mm2"])
        w.writerows(stat_rows)
    return summary


# ------------------------------------------------------------ 뇌 확대 비교 그림
def brain_zoom(rat, cfg):
    """뇌 중심부 확대 패널. PRE / 마지막 POST / (GRE면 FSE POST도) / 증강%.

    증강이 Gd 누출인지 PRE부터 있던 구조인지 눈으로 구분할 수 있게 한다.
    """
    OUT_DIR, ANIMAL = cfg["out"], cfg["label"]
    f = os.path.join(OUT_DIR, "corrected_volumes.npz")
    if not os.path.exists(f):
        return
    d = np.load(f)
    d_out = os.path.join(OUT_DIR, "brain_zoom")
    os.makedirs(d_out, exist_ok=True)
    for kind in KINDS:
        for ori in ("axial", "coronal"):
            kk = key_of(kind, ori, "PRE")
            if kk not in d.files:
                continue
            pre = d[kk].astype(float)
            posts = sorted([x for x in d.files
                            if x.startswith(key_of(kind, ori, "POST"))],
                           key=lambda x: int(re.search(r"POST(\d*)$", x).group(1) or 1))
            last = d[posts[-1]].astype(float)
            m = d[key_of(kind, ori, "analysis_mask")]
            enh = np.where(m, (last - pre) / (pre + 1e-6) * 100.0, np.nan)
            wl = np.percentile(pre[pre > 0], [2, 99.5])
            pan = [("PRE", pre, wl), (posts[-1].split("_")[-1], last, wl)]
            if kind == "gre":
                fk = key_of("fse", ori, "POST")
                if fk in d.files:
                    fv = zoom_to(d[fk].astype(float), pre.shape[1], pre.shape[2])
                    pan.append(("FSE POST", fv, np.percentile(fv[fv > 0], [2, 99.5])))
            idx = np.argwhere(m.any(0))
            r0, c0 = idx.min(0); r1, c1 = idx.max(0) + 1
            dr, dc = int((r1 - r0) * 0.22), int((c1 - c0) * 0.22)
            r0, r1, c0, c1 = r0 + dr, r1 - dr, c0 + dc, c1 - dc
            for z in range(pre.shape[0]):
                fig, ax = plt.subplots(1, len(pan) + 1,
                                       figsize=(4.0 * (len(pan) + 1), 4.6))
                for a, (nm, vv, ww) in zip(ax, pan):
                    a.imshow(vv[z, r0:r1, c0:c1], cmap="gray", vmin=ww[0], vmax=ww[1])
                    a.set_title(nm, fontsize=10)
                im = ax[-1].imshow(enh[z, r0:r1, c0:c1], cmap="jet", vmin=0, vmax=60)
                ax[-1].set_title("증강 %", fontsize=10)
                for a in ax:
                    a.set_xticks(np.arange(0, c1 - c0, 20))
                    a.set_yticks(np.arange(0, r1 - r0, 20))
                    a.set_xticklabels(np.arange(c0, c1, 20), fontsize=6)
                    a.set_yticklabels(np.arange(r0, r1, 20), fontsize=6)
                    a.grid(alpha=0.25, color="w", lw=0.4)
                fig.colorbar(im, ax=ax[-1], fraction=0.046, pad=0.03, label="% 증강")
                fig.suptitle(f"{ANIMAL}_{kind}_{ori}  slice {z} (0-index)  "
                             f"crop row {r0}~{r1} col {c0}~{c1}", fontsize=11)
                fig.tight_layout()
                fig.savefig(os.path.join(d_out,
                            f"{ANIMAL}_{kind}_{ori}_slice{z:02d}_zoom.png"),
                            dpi=120, bbox_inches="tight")
                plt.close(fig)
            print(f"  {ANIMAL}_{kind}_{ori}: 뇌 확대 {pre.shape[0]}장 저장", flush=True)


# ------------------------------------------------------------ 좌우 대칭 비교
def _mirror_op(rc, c0, th):
    """(행 rc, 열 c0)을 지나고 열 축에서 th(rad)만큼 기운 직선에 대한 반사.

    반환 (2x2 행렬 R, 오프셋). 반사는 자기 역변환이라 affine_transform에 그대로 넣으면
    된다. th=0이면 col' = 2*c0 - col 인 기존의 수직축 반사와 같다.
    """
    d = np.array([np.cos(th), np.sin(th)])
    R = 2 * np.outer(d, d) - np.eye(2)
    p0 = np.array([rc, c0])
    return R, p0 - R @ p0


def apply_mirror(vol, R, off, order=1):
    """반사를 볼륨에 적용(슬라이스 축은 건드리지 않는다)."""
    M = np.eye(3); M[1:, 1:] = R
    o = np.zeros(3); o[1:] = off
    return ndimage.affine_transform(vol, M, offset=o, order=order,
                                    mode="constant", cval=0.0)


def mirror_axis(vol, mask):
    """대칭 직선을 **기울기와 위치 둘 다** 맞춰 찾는다. 반환 (rc, c0, theta, 상관).

    26-08-26까지는 평행이동만 맞췄다. 그러면 머리가 기운 개체에서 정렬 오차가
    좌우 차이로 새어나와, 뇌 전체에 넓은 빨강/파랑 띠가 생긴다(0911 개체 4에서
    실제로 그랬다). 기울기를 같이 맞추면 이 띠가 사라진다.

    탐색은 다운샘플 볼륨에서 거칠게 하고 같은 해상도에서 미세 조정한다. 축을 멀리
    옮기면 겹침이 줄어 상관이 허수로 올라가므로, 겹침이 마스크의 75% 미만이면 버린다.
    상관이 낮으면 좌우 비교가 그만큼 부정확하다는 뜻이니 결과에 함께 적는다.
    """
    idx = np.argwhere(mask)
    rc, cen = float(idx[:, 1].mean()), float(idx[:, 2].mean())
    ds = MIRROR_DS
    vs = np.ascontiguousarray(vol[:, ::ds, ::ds])
    ms = np.ascontiguousarray(mask[:, ::ds, ::ds])
    need = 0.75 * ms.sum()
    rcs, cens = rc / ds, cen / ds

    def score(c0, th):
        R, off = _mirror_op(rcs, c0, th)
        g = apply_mirror(vs, R, off)
        gm = apply_mirror(ms.astype(np.float32), R, off, order=0) > 0.5
        k = ms & gm
        if k.sum() < need:
            return None
        return float(np.corrcoef(vs[k], g[k])[0, 1])

    best = (cens, 0.0, -1.0)
    for th in np.radians(np.arange(-MIRROR_MAX_DEG, MIRROR_MAX_DEG + 1e-9, 2.0)):
        for c0 in np.arange(cens - 20 / ds, cens + 20 / ds + 1e-9, 2.0 / ds):
            r = score(c0, th)
            if r is not None and r > best[2]:
                best = (float(c0), float(th), r)
    c0b, thb = best[0], best[1]
    for th in thb + np.radians(np.arange(-1.5, 1.51, 0.25)):
        for c0 in np.arange(c0b - 2.0 / ds, c0b + 2.0 / ds + 1e-9, 0.25 / ds):
            r = score(c0, th)
            if r is not None and r > best[2]:
                best = (float(c0), float(th), r)
    return rc, best[0] * ds, best[1], best[2]


_AXIS_CACHE = {}


def _asym(d, kind, ori, tag=""):
    """(증강% 좌우 비대칭 맵, 양쪽 유효 마스크, 반사 파라미터, 좌우 상관).

    대칭축 피팅은 느리고 같은 슬랩에서 세 번 불리므로 (tag, kind, ori)로 캐시한다.
    """
    pre = d[key_of(kind, ori, "PRE")].astype(float)
    posts = sorted([x for x in d.files if x.startswith(key_of(kind, ori, "POST"))],
                   key=lambda x: int(re.search(r"POST(\d*)$", x).group(1) or 1))
    last = d[posts[-1]].astype(float)
    m = d[key_of(kind, ori, "analysis_mask")]
    ck = (tag, kind, ori)
    if ck not in _AXIS_CACHE:
        _AXIS_CACHE[ck] = mirror_axis(pre, m)
    rc, c0, th, lr = _AXIS_CACHE[ck]
    R, off = _mirror_op(rc, c0, th)
    enh = np.where(m, (last - pre) / (pre + 1e-6) * 100.0, 0.0)
    emir = apply_mirror(enh, R, off)
    mmir = apply_mirror(m.astype(np.float32), R, off, order=0) > 0.5
    both = m & mmir
    asym = ndimage.gaussian_filter(np.where(both, enh - emir, 0.0), (0, 1.5, 1.5))
    return asym, both, (rc, c0, th, R, off), lr


def mirror_asym(rat, cfg):
    """좌우 거울상 차이로 편측 증강을 본다(대측 반구가 곧 내부 대조군).

    FUS-BBBO는 보통 한쪽만 연다. 두피·측두근은 좌우가 대체로 대칭이므로 거울상
    차이를 보면 뇌 분할 없이도 편측 성분만 남길 수 있다. 남는 한계: 머리가 기울면
    정렬 오차가 차이로 나타나고, 원래 좌우 비대칭인 구조(혈관 등)도 남는다.
    """
    OUT_DIR, ANIMAL = cfg["out"], cfg["label"]
    f = os.path.join(OUT_DIR, "corrected_volumes.npz")
    if not os.path.exists(f):
        return
    d = np.load(f)
    d_out = os.path.join(OUT_DIR, "mirror_asym")
    os.makedirs(d_out, exist_ok=True)
    rows = []
    for kind in KINDS:
        for ori in ("axial", "coronal"):
            if key_of(kind, ori, "PRE") not in d.files:
                continue
            asym, both, mp, lr = _asym(d, kind, ori, ANIMAL)
            rc, c0, th = mp[0], mp[1], mp[2]
            posts = sorted([x for x in d.files
                            if x.startswith(key_of(kind, ori, "POST"))],
                           key=lambda x: int(re.search(r"POST(\d*)$", x).group(1) or 1))
            last = d[posts[-1]].astype(float)
            sp = d[key_of(kind, ori, "spacing")]
            print(f"  {ANIMAL}_{kind}_{ori}: 대칭축 col={c0:.2f} "
                  f"기울기 {np.degrees(th):+.2f}도 (좌우 상관 {lr:.3f}), "
                  f"양쪽 유효 화소 {int(both.sum())}", flush=True)
            dasym = np.where(both, asym, np.nan)
            M = float(np.nanpercentile(np.abs(dasym), 99)) or 1.0
            wl = np.percentile(last[last > 0], [2, 99])
            n = last.shape[0]; cols = min(7, n); rr = int(np.ceil(n / cols))
            fig, axes = plt.subplots(rr, cols, figsize=(2.6 * cols, 2.8 * rr), squeeze=False)
            axes = axes.ravel(); im = None
            for z in range(n):
                axes[z].imshow(last[z], cmap="gray", vmin=wl[0], vmax=wl[1])
                im = axes[z].imshow(dasym[z], cmap="seismic", vmin=-M, vmax=M, alpha=0.6)
                axes[z].set_title(f"slice {z}", fontsize=8)
            for a in axes:
                a.axis("off")
            fig.suptitle(f"{ANIMAL}_{kind}_{ori}  증강% 좌우 거울상 차이 "
                         f"(자기 - 반대편), 대칭축 col={c0:.1f} "
                         f"기울기 {np.degrees(th):+.1f}도", fontsize=12)
            fig.subplots_adjust(right=0.9)
            fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.010, 0.7]), label="%p")
            fig.savefig(os.path.join(d_out, f"{ANIMAL}_{kind}_{ori}_mirror_asym.png"),
                        dpi=130, bbox_inches="tight")
            plt.close(fig)
            vox = abs(sp[0] * sp[1])
            for z in range(n):
                a = dasym[z][both[z]]
                if a.size < 100:
                    continue
                rows.append([kind, ori, z, round(c0, 2), round(float(np.degrees(th)), 2),
                             round(lr, 3), int(both[z].sum()),
                             round(float(np.median(a)), 2),
                             round(float(np.percentile(a, 95)), 2),
                             int((a > 10).sum()), round(float((a > 10).sum()) * vox, 2)])
    if rows:
        with open(os.path.join(d_out, "mirror_asym_slice_stats.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["kind", "orientation", "slice_0idx", "mirror_axis_col",
                        "mirror_tilt_deg", "lr_corr",
                        "paired_vox", "asym_median_pp", "asym_p95_pp",
                        "vox_asym_gt10pp", "area_gt10pp_mm2"])
            w.writerows(rows)


# ------------------------------------------------------------ 편측 병소 정량
def lateral_focus(rat, cfg):
    """좌우 비대칭이 큰 덩어리를 찾아 대측 거울상 ROI와 정량 비교한다.

    순 누출 = 병소 증강% - 대측 증강%. `mri_dce.bbbo_metrics`와 같은 정의이고
    15%p 이상이면 양성으로 본다.

    한계를 분명히 해 둔다. 해부학적 뇌 분할이 없다. 두피·근육을 빼려고 조직 마스크를
    21화소 침식했을 뿐이라 목덜미 근육이나 영상 가장자리 덩어리가 후보에 남는다.
    `mirror_enh_last_pct`로 거른다. 정상 뇌실질 대측은 10% 아래, 근육이면 25%를 넘는다.
    대칭축은 평행이동만 맞춘다. 머리가 기울면 그만큼 오차가 남는다.
    """
    OUT_DIR, ANIMAL = cfg["out"], cfg["label"]
    f = os.path.join(OUT_DIR, "corrected_volumes.npz")
    if not os.path.exists(f):
        return []
    d = np.load(f)
    d_out = os.path.join(OUT_DIR, "lateral_focus")
    os.makedirs(d_out, exist_ok=True)
    rows = []
    for kind in KINDS:
        for ori in ("axial", "coronal"):
            if key_of(kind, ori, "PRE") not in d.files:
                continue
            asym, both, mp, lr = _asym(d, kind, ori, ANIMAL)
            rc, c0, th, R, off = mp
            tps = ["PRE"] + sorted(
                [x.split("_")[-1] for x in d.files
                 if x.startswith(key_of(kind, ori, "POST"))],
                key=lambda x: int(re.search(r"POST(\d*)", x).group(1) or 1))
            vols = [d[key_of(kind, ori, t)].astype(float) for t in tps]
            pre, last = vols[0], vols[-1]
            sp = d[key_of(kind, ori, "spacing")]
            N = pre.shape[2]
            inner = ndimage.binary_erosion(both, np.ones((1, FOCUS_ERODE, FOCUS_ERODE)))
            cand = ndimage.binary_opening(inner & (asym > FOCUS_ASYM_PP), np.ones((1, 3, 3)))
            lab, n = ndimage.label(cand)
            if n == 0:
                print(f"  {ANIMAL}_{kind}_{ori}: {FOCUS_ASYM_PP}%p 넘는 편측 덩어리 없음",
                      flush=True)
                continue
            sizes = ndimage.sum(cand, lab, range(1, n + 1))
            vox = abs(sp[0] * sp[1] * sp[2])
            for rank, k in enumerate(np.argsort(sizes)[::-1][:3], 1):
                if sizes[k] < FOCUS_MIN_VOX:
                    break
                sel = lab == (k + 1)
                # 대측 ROI = 병소 ROI를 같은 반사로 옮긴 것. 축이 기울어도 맞는다.
                mir = apply_mirror(sel.astype(np.float32), R, off, order=0) > 0.5
                zc, rc2, cc = np.argwhere(sel).mean(0)
                sig = [float(v[sel].mean()) for v in vols]
                sgm = [float(v[mir].mean()) for v in vols]
                e = [(x - sig[0]) / sig[0] * 100.0 for x in sig]
                em = [(x - sgm[0]) / sgm[0] * 100.0 for x in sgm]
                net = e[-1] - em[-1]
                zs = sorted({int(z_) for z_ in np.argwhere(sel)[:, 0]})
                # 병소 PRE가 낮으면 %증강은 분모 때문에 부풀려진다. 분모 없는 dS로
                # 교차 확인할 수 있어야 한다.
                dsf, dsm = sig[-1] - sig[0], sgm[-1] - sgm[0]
                edge = "예" if (min(zs) == 0 or max(zs) == pre.shape[0] - 1) else "아니오"
                rows.append([kind, ori, rank, int(sizes[k]), round(sizes[k] * vox, 2),
                             round(c0, 2), round(float(np.degrees(th)), 2),
                             round(lr, 3), edge, round(zc, 1),
                             int(round(rc2)), int(round(cc)),
                             " ".join(str(z_) for z_ in zs),
                             round(float(asym[sel].mean()), 1),
                             round(float(asym[sel].max()), 1),
                             " ".join(f"{x:+.1f}" for x in e),
                             " ".join(f"{x:+.1f}" for x in em),
                             round(net, 1), "양성" if net >= 15 else "음성",
                             round(em[-1], 1), round(sig[0], 1), round(sgm[0], 1),
                             round(sig[0] / sgm[0], 3),
                             round(dsf, 1), round(dsm, 1), round(dsf - dsm, 1)])
                print(f"  {ANIMAL}_{kind}_{ori} #{rank}: {int(sizes[k])}화소 "
                      f"{sizes[k]*vox:.2f}mm3 중심(z {zc:.1f}, row {rc2:.0f}, col {cc:.0f}) "
                      f"슬라이스 {zs}  순 누출 {net:+.1f}%p  대측최종 {em[-1]:+.1f}%  "
                      f"PRE비 {sig[0]/sgm[0]:.2f}", flush=True)

                wl = np.percentile(pre[pre > 0], [2, 99.5])
                for z_ in zs:
                    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.5))
                    for a, (nm, vv) in zip(ax, [("PRE", pre), (tps[-1], last)]):
                        a.imshow(vv[z_], cmap="gray", vmin=wl[0], vmax=wl[1])
                        a.contour(sel[z_], levels=[0.5], colors="r", linewidths=0.9)
                        a.contour(mir[z_], levels=[0.5], colors="c", linewidths=0.9)
                        a.set_title(nm, fontsize=10); a.axis("off")
                    im = ax[2].imshow(asym[z_], cmap="seismic", vmin=-60, vmax=60)
                    ax[2].contour(sel[z_], levels=[0.5], colors="k", linewidths=0.9)
                    ax[2].set_title("좌우 비대칭 %p", fontsize=10); ax[2].axis("off")
                    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.03, label="%p")
                    fig.suptitle(f"{ANIMAL}_{kind}_{ori} 병소#{rank} slice {z_} — "
                                 f"빨강=병소 ROI, 하늘=대측 거울 ROI, "
                                 f"순 누출 {net:+.1f}%p", fontsize=11)
                    fig.tight_layout()
                    fig.savefig(os.path.join(
                        d_out, f"{ANIMAL}_{kind}_{ori}_focus{rank}_slice{z_:02d}.png"),
                        dpi=125, bbox_inches="tight")
                    plt.close(fig)

                fig, a = plt.subplots(figsize=(5.6, 4.2))
                x = np.arange(len(tps))
                a.plot(x, e, "o-", color="crimson", label="병소")
                a.plot(x, em, "s-", color="steelblue", label="대측 거울상")
                a.axhline(0, color="k", lw=0.6, ls=":")
                a.set_xticks(x); a.set_xticklabels(tps, fontsize=8)
                a.set_ylabel("증강 % (PRE 기준)")
                a.set_title(f"{ANIMAL}_{kind}_{ori} 병소#{rank}  순 누출 {net:+.1f}%p",
                            fontsize=11)
                a.legend(); a.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(
                    d_out, f"{ANIMAL}_{kind}_{ori}_focus{rank}_curve.png"), dpi=130)
                plt.close(fig)
    if rows:
        with open(os.path.join(d_out, "lateral_focus.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            # edge_slice=예 면 첫/마지막 슬라이스에 걸친 후보다. 정합 가장자리
            # 인공물일 가능성이 높으니 그림으로 확인할 것.
            w.writerow(["kind", "orientation", "rank", "voxels", "volume_mm3",
                        "mirror_axis_col", "mirror_tilt_deg", "lr_corr", "edge_slice",
                        "center_z", "center_row",
                        "center_col", "slices", "asym_mean_pp", "asym_max_pp",
                        "focus_enh_pct_by_frame", "mirror_enh_pct_by_frame",
                        "net_leakage_pp_lastPOST", "verdict_15pp",
                        "mirror_enh_last_pct", "PRE_focus", "PRE_mirror",
                        "PRE_focus_over_mirror", "dS_focus", "dS_mirror", "dS_diff"])
            w.writerows(rows)
    return rows


# ------------------------------------------------------------ 국소 대비 병소 검출
LOCAL_BG_SIGMA = 20     # 국소 배경 추정 가우시안 시그마 (화소). 약 2.3mm.
LOCAL_EXCESS_PP = 15    # 국소 배경 대비 초과 문턱 (%p)
LOCAL_MIN_VOX = 40      # 최소 화소수
SHELL_IN, SHELL_OUT = 4, 12   # 대조 고리 반경 (화소). 병소에서 0.5~1.4mm 떨어진 띠.


def _disk(rr):
    y, x = np.mgrid[-rr:rr + 1, -rr:rr + 1]
    return (y ** 2 + x ** 2 <= rr * rr)[None, :, :]


def local_focus(rat, cfg):
    """좌우 대칭을 가정하지 않고 국소 증강 덩어리를 찾는다.

    왜 필요한가. 거울상 비대칭법은 한쪽만 열린다는 전제를 쓴다. 이 날짜는 한 뇌에
    여러 부위를 표적했으므로 양쪽을 같이 열었다면 거울상 차감에서 서로 상쇄돼 둘 다
    놓친다. 여기서는 대조군을 대측 반구가 아니라 **병소 주변 고리**로 잡는다.

    절차: 증강% 맵에서 마스크 가중 가우시안(시그마 20화소)으로 매끄러운 국소 배경을
    만들어 빼고 초과분을 얻는다. 조직 마스크를 21화소 침식해 두피·근육을 걷어낸 뒤
    초과분이 15%p를 넘는 덩어리를 찾는다. 각 덩어리마다 4~12화소 떨어진 고리를 같은
    슬라이스 안에서 대조 ROI로 잡아 순 누출을 계산한다.

    한계는 거울상법과 같은 것이 남는다. 해부학적 뇌 분할이 아니라 침식이고, 정합
    잔차가 조직 경계에 만드는 테두리가 국소 초과로 잡힐 수 있다. 그래서 edge_slice,
    shell_enh_last_pct(고리가 25%를 넘으면 근육)를 같이 낸다.
    """
    OUT_DIR, ANIMAL = cfg["out"], cfg["label"]
    f = os.path.join(OUT_DIR, "corrected_volumes.npz")
    if not os.path.exists(f):
        return []
    d = np.load(f)
    d_out = os.path.join(OUT_DIR, "local_focus")
    os.makedirs(d_out, exist_ok=True)
    rows = []
    for kind in KINDS:
        for ori in ("axial", "coronal"):
            if key_of(kind, ori, "PRE") not in d.files:
                continue
            tps = ["PRE"] + sorted(
                [x.split("_")[-1] for x in d.files
                 if x.startswith(key_of(kind, ori, "POST"))],
                key=lambda x: int(re.search(r"POST(\d*)", x).group(1) or 1))
            vols = [d[key_of(kind, ori, t)].astype(float) for t in tps]
            pre, last = vols[0], vols[-1]
            m = d[key_of(kind, ori, "analysis_mask")]
            sp = d[key_of(kind, ori, "spacing")]
            enh = np.where(m, (last - pre) / (pre + 1e-6) * 100.0, 0.0)
            num = ndimage.gaussian_filter(enh, (0, LOCAL_BG_SIGMA, LOCAL_BG_SIGMA))
            den = ndimage.gaussian_filter(m.astype(float), (0, LOCAL_BG_SIGMA, LOCAL_BG_SIGMA))
            bg = num / np.maximum(den, 1e-6)
            exc = ndimage.gaussian_filter(np.where(m, enh - bg, 0.0), (0, 1.5, 1.5))
            inner = ndimage.binary_erosion(m, np.ones((1, FOCUS_ERODE, FOCUS_ERODE)))
            cand = ndimage.binary_opening(inner & (exc > LOCAL_EXCESS_PP),
                                          np.ones((1, 3, 3)))
            lab, n = ndimage.label(cand)
            if n:
                sizes = ndimage.sum(cand, lab, range(1, n + 1))
            else:
                sizes = np.array([])
                print(f"  {ANIMAL}_{kind}_{ori}: 국소 초과 {LOCAL_EXCESS_PP}%p 넘는 덩어리 없음",
                      flush=True)
            vox = abs(sp[0] * sp[1] * sp[2])
            din, dout = _disk(SHELL_IN), _disk(SHELL_OUT)
            for rank, k in enumerate(np.argsort(sizes)[::-1][:4], 1):
                if sizes[k] < LOCAL_MIN_VOX:
                    break
                sel = lab == (k + 1)
                shell = (ndimage.binary_dilation(sel, dout)
                         & ~ndimage.binary_dilation(sel, din) & inner)
                if shell.sum() < 50:
                    continue
                zc, rc, cc = np.argwhere(sel).mean(0)
                sig = [float(v[sel].mean()) for v in vols]
                sgs = [float(v[shell].mean()) for v in vols]
                e = [(x - sig[0]) / sig[0] * 100.0 for x in sig]
                es = [(x - sgs[0]) / sgs[0] * 100.0 for x in sgs]
                net = e[-1] - es[-1]
                zs = sorted({int(z_) for z_ in np.argwhere(sel)[:, 0]})
                edge = "예" if (min(zs) == 0 or max(zs) == pre.shape[0] - 1) else "아니오"
                dsf, dss = sig[-1] - sig[0], sgs[-1] - sgs[0]
                rows.append([kind, ori, rank, int(sizes[k]), round(sizes[k] * vox, 2),
                             round(zc, 1), int(round(rc)), int(round(cc)),
                             " ".join(str(z_) for z_ in zs), edge,
                             round(float(exc[sel].mean()), 1),
                             round(float(exc[sel].max()), 1),
                             " ".join(f"{x:+.1f}" for x in e),
                             " ".join(f"{x:+.1f}" for x in es),
                             round(net, 1), "양성" if net >= 15 else "음성",
                             round(es[-1], 1), round(sig[0], 1), round(sgs[0], 1),
                             round(sig[0] / sgs[0], 3),
                             round(dsf, 1), round(dss, 1), round(dsf - dss, 1)])
                print(f"  {ANIMAL}_{kind}_{ori} L#{rank}: {int(sizes[k])}화소 "
                      f"{sizes[k]*vox:.2f}mm3 중심(z {zc:.1f}, row {rc:.0f}, col {cc:.0f}) "
                      f"슬라이스 {zs}  순 누출 {net:+.1f}%p  고리 {es[-1]:+.1f}%  "
                      f"가장자리={edge}  PRE비 {sig[0]/sgs[0]:.2f}", flush=True)

                wl = np.percentile(pre[pre > 0], [2, 99.5])
                for z_ in zs:
                    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.5))
                    for a, (nm, vv) in zip(ax, [("PRE", pre), (tps[-1], last)]):
                        a.imshow(vv[z_], cmap="gray", vmin=wl[0], vmax=wl[1])
                        a.contour(sel[z_], levels=[0.5], colors="r", linewidths=0.9)
                        a.contour(shell[z_], levels=[0.5], colors="c", linewidths=0.7)
                        a.set_title(nm, fontsize=10)
                        a.axis("off")
                    im = ax[2].imshow(exc[z_], cmap="inferno", vmin=0, vmax=40)
                    ax[2].contour(sel[z_], levels=[0.5], colors="c", linewidths=0.9)
                    ax[2].set_title("국소 배경 대비 초과 %p", fontsize=10)
                    ax[2].axis("off")
                    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.03, label="%p")
                    fig.suptitle(f"{ANIMAL}_{kind}_{ori} 국소병소#{rank} slice {z_} — "
                                 f"빨강=병소, 하늘=주변 고리 대조, 순 누출 {net:+.1f}%p",
                                 fontsize=11)
                    fig.tight_layout()
                    fig.savefig(os.path.join(
                        d_out, f"{ANIMAL}_{kind}_{ori}_local{rank}_slice{z_:02d}.png"),
                        dpi=125, bbox_inches="tight")
                    plt.close(fig)

                fig, a = plt.subplots(figsize=(5.6, 4.2))
                x = np.arange(len(tps))
                a.plot(x, e, "o-", color="crimson", label="병소")
                a.plot(x, es, "s-", color="steelblue", label="주변 고리")
                a.axhline(0, color="k", lw=0.6, ls=":")
                a.set_xticks(x)
                a.set_xticklabels(tps, fontsize=8)
                a.set_ylabel("증강 % (PRE 기준)")
                a.set_title(f"{ANIMAL}_{kind}_{ori} 국소병소#{rank}  순 누출 {net:+.1f}%p",
                            fontsize=11)
                a.legend()
                a.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(os.path.join(
                    d_out, f"{ANIMAL}_{kind}_{ori}_local{rank}_curve.png"), dpi=130)
                plt.close(fig)

            M = float(np.percentile(exc[inner], 99.5)) if inner.sum() else 40.0
            nz = pre.shape[0]
            cols = min(7, nz)
            rr = int(np.ceil(nz / cols))
            fig, axes = plt.subplots(rr, cols, figsize=(2.6 * cols, 2.8 * rr), squeeze=False)
            axes = axes.ravel()
            im = None
            for z_ in range(nz):
                im = axes[z_].imshow(np.where(inner[z_], exc[z_], np.nan),
                                     cmap="inferno", vmin=0, vmax=max(M, 20))
                axes[z_].set_title(f"slice {z_}", fontsize=8)
            for a in axes:
                a.axis("off")
            fig.suptitle(f"{ANIMAL}_{kind}_{ori}  국소 배경 대비 초과 증강 "
                         f"(배경 시그마 {LOCAL_BG_SIGMA}화소, 침식 후 영역만)", fontsize=12)
            fig.subplots_adjust(right=0.9)
            fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.010, 0.7]), label="%p")
            fig.savefig(os.path.join(d_out, f"{ANIMAL}_{kind}_{ori}_excess_montage.png"),
                        dpi=130, bbox_inches="tight")
            plt.close(fig)
    if rows:
        with open(os.path.join(d_out, "local_focus.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            # shell_enh_last_pct 가 25%를 넘으면 ROI가 뇌가 아니라 두피/근육이다.
            w.writerow(["kind", "orientation", "rank", "voxels", "volume_mm3",
                        "center_z", "center_row", "center_col", "slices", "edge_slice",
                        "excess_mean_pp", "excess_max_pp",
                        "focus_enh_pct_by_frame", "shell_enh_pct_by_frame",
                        "net_leakage_pp_lastPOST", "verdict_15pp", "shell_enh_last_pct",
                        "PRE_focus", "PRE_shell", "PRE_focus_over_shell",
                        "dS_focus", "dS_shell", "dS_diff"])
            w.writerows(rows)
    return rows


# ------------------------------------------------------------ 직교 방향 교차 확인
def _ori_geom(rat_dir):
    """(계열, 방향) → (첫 슬라이스 IPP, r, c, n, 화소크기, 슬라이스 간격). PRE 기준."""
    out = {}
    for folder in sorted(os.listdir(rat_dir)):
        d = os.path.join(rat_dir, folder)
        if not os.path.isdir(d):
            continue
        kind, tp = classify(folder)
        if kind == "other" or tp != 0:
            continue
        fs = sorted(glob.glob(os.path.join(d, "*.dcm")))
        ds = [pydicom.dcmread(f, stop_before_pixels=True) for f in fs]
        iop = np.array([float(x) for x in ds[0].ImageOrientationPatient])
        r, c = iop[:3], iop[3:]
        n = np.cross(r, c)
        ds.sort(key=lambda x: float(np.dot(
            np.array([float(v) for v in x.ImagePositionPatient]), n)))
        ipp = np.array([float(v) for v in ds[0].ImagePositionPatient])
        ps = float(ds[0].PixelSpacing[0])
        sz = float(getattr(ds[0], "SpacingBetweenSlices", None) or ds[0].SliceThickness)
        out[(kind, classify_orientation(iop))] = (ipp, r, c, n, ps, sz)
    return out


def cross_orientation_check(rat, cfg, rows):
    """한 방향에서 찾은 병소를 같은 계열의 직교 방향 같은 물리 좌표에서 다시 본다.

    따로 찍은 영상에서 같은 자리에 나오면 정합·잡음 탓으로 보기 어렵다.
    비교 기준은 그 슬라이스 뇌 안쪽 비대칭 분포의 p95/p99다.
    """
    if not rows:
        return
    OUT_DIR, DATA_DIR, ANIMAL = cfg["out"], cfg["data"], cfg["label"]
    d = np.load(os.path.join(OUT_DIR, "corrected_volumes.npz"))
    G = _ori_geom(DATA_DIR)
    A = {}
    for kind in KINDS:
        for ori in ("axial", "coronal"):
            if key_of(kind, ori, "PRE") not in d.files:
                continue
            asym, both, _, _ = _asym(d, kind, ori, ANIMAL)
            A[(kind, ori)] = (asym, ndimage.binary_erosion(
                both, np.ones((1, FOCUS_ERODE, FOCUS_ERODE))))
    out = []
    for x in rows:
        kind, src = x[0], x[1]
        dst = "coronal" if src == "axial" else "axial"
        if (kind, src) not in G or (kind, dst) not in A or (kind, dst) not in G:
            continue
        ipp, r, c, n, ps, sz = G[(kind, src)]
        P = ipp + x[11] * ps * r + x[10] * ps * c + x[9] * sz * n
        ipd, rd, cd, nd, psd, szd = G[(kind, dst)]
        dd = P - ipd
        zi = int(round(float(np.dot(dd, nd)) / szd))
        ri = int(round(float(np.dot(dd, cd)) / psd))
        ci = int(round(float(np.dot(dd, rd)) / psd))
        asym, inner = A[(kind, dst)]
        if not (0 <= zi < asym.shape[0] and 0 <= ri < asym.shape[1]
                and 0 <= ci < asym.shape[2]):
            out.append([ANIMAL, kind, src, x[2], dst, zi, ri, ci, "", "", "", "", "격자 밖"])
            continue
        patch = asym[zi, max(0, ri - 4):ri + 5, max(0, ci - 4):ci + 5]
        v = asym[zi][inner[zi]]
        med = float(np.median(patch))
        p95 = float(np.percentile(v, 95)) if v.size else float("nan")
        p99 = float(np.percentile(v, 99)) if v.size else float("nan")
        out.append([ANIMAL, kind, src, x[2], dst, zi, ri, ci, round(med, 1),
                    round(float(patch.max()), 1), round(p95, 1), round(p99, 1),
                    "재현" if med >= p99 else ("경계" if med >= p95 else "미재현")])
    if out:
        with open(os.path.join(OUT_DIR, "lateral_focus", "crosscheck.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["animal", "kind", "src_orientation", "rank", "dst_orientation",
                        "dst_z", "dst_row", "dst_col", "dst_asym_median_pp",
                        "dst_asym_max_pp", "dst_slice_p95_pp", "dst_slice_p99_pp",
                        "verdict"])
            w.writerows(out)
        for o in out:
            print(f"  {ANIMAL} {o[1]} {o[2]}#{o[3]} -> {o[4]} "
                  f"z={o[5]} row={o[6]} col={o[7]}: 비대칭 {o[8]}%p "
                  f"(p95 {o[10]}, p99 {o[11]}) -> {o[-1]}", flush=True)


# ------------------------------------------------------------ 실행
argv = sys.argv[1:]
FIGS_ONLY = "figs" in argv
keys = [a for a in argv if a != "figs"] or list(SETS)
bad = [k for k in keys if k not in SETS]
if bad:
    sys.exit(f"알 수 없는 키: {bad}\n사용 가능: {list(SETS)}")

allsum = []
for k in keys:
    if not FIGS_ONLY:
        allsum += run_one(k, SETS[k])
    else:
        print(f"\n########## RAT {k} (그림·정량만) ##########", flush=True)
    brain_zoom(k, SETS[k])
    mirror_asym(k, SETS[k])
    cross_orientation_check(k, SETS[k], lateral_focus(k, SETS[k]))
    local_focus(k, SETS[k])

if allsum:
    with open(os.path.join(OUT_ROOT, "enhancement_summary_all.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rat", "kind", "orientation", "n_frames", "roi_mm3",
                    "signal_by_frame", "enh_pct_by_frame", "peak_pct",
                    "time_to_peak_frame", "washin", "auc"])
        w.writerows(allsum)
    print("\n\n===== 전체 요약 =====", flush=True)
    print("%-4s %-4s %-9s %6s  %s" % ("rat", "kind", "ori", "peak%", "enh% by frame"))
    for r in allsum:
        print("%-4s %-4s %-9s %6s  %s" % (r[0], r[1], r[2], r[7], r[6]))
print("\n완료", flush=True)
