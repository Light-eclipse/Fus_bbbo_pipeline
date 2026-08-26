# -*- coding: utf-8 -*-
"""260826 CORN BBBO — 시리즈별 강도 스케일 보정 DCE 분석 (개체 1, 2).

이전 날짜들과 같은 절차다. Agilent FdfToDcm이 시리즈(=시점)마다 화소값을 따로
정규화해 저장했고, 그 인자가 사설 태그 (00E1,1001)에 있다. `참값 = 화소값 / 태그`로
되돌려야 PRE와 POST를 같은 축에서 비교할 수 있다. 보정 검증은 Gd·수신이득과 무관한
공기 배경 잡음으로 하고, 배경 마스크는 **보정 후** 데이터의 하위 10%로 잡는다.

이 데이터에만 있는 사정
----------------------
1. DCE 계열(PRE, POST1~POST5)은 개체·방향별로 TR/TE/FA/NEX가 완전히 동일하다.
   PRE와 POST의 IPP도 소수점까지 같다(격자 차 0.0000mm). 같은 세션 연속 촬영이다.
2. `GRE_*_POST_TR200_12`는 TR 200 / FA 60 / NEX 2로 DCE 계열(TR 78~111 / FA 40~45 /
   NEX 4)과 다른 프로토콜이다. `FSE_T1_*`은 fsems(TR 750 / TE 27.7 / FA 90 / 192²)로
   시퀀스 자체가 다르고 Gd 전 대응 영상이 없다.
   → 두 계열은 DCE 증강 곡선에 넣지 않는다. 스케일 보정만 해서 따로 낸다.
3. 개체 2의 `GRE_COR_POST_TR200_12_NO2`는 헤더 IPP/IOP/TE가 개체 1의 COR과 같다.
   개체 1의 처방을 그대로 불러 촬영한 것으로 보인다. 개체 2 자신의 영상은 맞다:
   개체 2 격자로 되돌리면 개체 2 COR POST5와의 상관이 0.852 → 0.981로 올라가고
   (개체 1의 같은 쌍은 0.990), 같은 격자에 놓고도 개체 1 POST5와는 0.824에 그친다.
   → 기하 리샘플로 개체 2 격자에 되돌려서 쓴다.

실행
----
    python "_run_0826.py"          # 개체 1, 2 전부
    python "_run_0826.py" 1        # 개체 1만
    python "_run_0826.py" figs     # 저장된 npz로 brain_zoom 그림만 다시 만들기
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
BASE = os.path.join(ROOT, r"26-08-26")
DATA_ROOT = os.path.join(BASE, "260826_CORN_BBBO")
OUT_ROOT = os.path.join(BASE, "2. 분석 결과")

SCALE_TAG = (0x00E1, 0x1001)
AIR_PCT = 10           # 배경 마스크 백분위수
HEAT_VMAX = 100        # 슬라이스별 히트맵 컬러 상한 %
HEAT_VMAX_BRAIN = 40   # 뇌실질 감도용 몽타주 상한 %

SETS = {
    "1": dict(data=os.path.join(DATA_ROOT, "1"),
              out=os.path.join(OUT_ROOT, "RAT 1"), label="0826_NO1"),
    "2": dict(data=os.path.join(DATA_ROOT, "2"),
              out=os.path.join(OUT_ROOT, "RAT 2"), label="0826_NO2"),
}

sys.path.insert(0, CODE_DIR)
from mri_dce import (_load_volumes, Slab, register_slab, enhancement_curve,
                     classify_orientation,
                     plot_curves, save_frame_slices, heatmap_frames_row,
                     subtraction_frames_row, montage_anatomy, montage_heatmap)


# ------------------------------------------------------------ 시리즈 분류
def classify(folder):
    """폴더명 → (계열, 시점). 계열: 'dce' | 'tr200' | 'fse'."""
    f = folder.upper()
    if "FSE_T1" in f:
        return "fse", None
    if "TR200" in f:
        return "tr200", None
    m = re.search(r"GRE_(?:AX|COR)_POST(\d+)_", f)
    if m:
        return "dce", int(m.group(1))
    if re.search(r"GRE_(?:AX|COR)_PRE_", f):
        return "dce", 0
    return "other", None


def series_meta(rat_dir):
    """시리즈 UID → 최소투영 슬라이스 IPP / 스케일 인자 / 폴더명 / 계열 / 시점."""
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
                    print(f"  [경고] {os.path.basename(root)}: 스케일 태그 없음 -> 1.0", flush=True)
                name_out[uid] = os.path.basename(root)
    return ({uid: ipp for uid, (p, ipp) in ipp_out.items()}, scale_out, name_out)


def series_params(rat_dir, folder):
    fs = sorted(glob.glob(os.path.join(rat_dir, folder, "*.dcm")))
    ds = pydicom.dcmread(fs[0], stop_before_pixels=True)
    return dict(n=len(fs), seq=str(ds.get("SequenceName", "?")),
                tr=float(ds.RepetitionTime), te=float(ds.EchoTime),
                fa=float(ds.FlipAngle), nex=float(ds.NumberOfAverages),
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


# ------------------------------------------------------------ 개체 1개 처리
def run_one(key, cfg):
    DATA_DIR, OUT_DIR, ANIMAL = cfg["data"], cfg["out"], cfg["label"]
    print(f"\n########## RAT {key}  ({ANIMAL}) ##########", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    vols_raw = _load_volumes(DATA_DIR, True)
    ippmin, scale, sname = series_meta(DATA_DIR)

    # 스케일 보정 + 계열/시점 재지정
    # (_tp_rank는 이 데이터에 못 쓴다. 'POST_TR200'이 POST1과 같은 tp=1로 파싱된다.)
    vols = []
    for v in vols_raw:
        kind, tp = classify(sname[v.series_uid])
        vols.append((kind, replace(v, data=v.data / scale[v.series_uid], tp=tp)))

    dce = [v for k, v in vols if k == "dce"]
    aux = [(k, v) for k, v in vols if k in ("tr200", "fse")]
    if not dce:
        print("  [경고] DCE 시리즈 없음 -> 건너뜀", flush=True)
        return []

    # --- 스캔 프로토콜 표
    with open(os.path.join(OUT_DIR, "series_protocol.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["kind", "orientation", "timepoint", "series_folder", "content_time",
                    "sequence", "TR_ms", "TE_ms", "flip_deg", "NEX", "matrix",
                    "pixel_mm", "n_slices", "scale_tag_00E1_1001"])
        for k, v in sorted(vols, key=lambda x: sname[x[1].series_uid]):
            if k == "other":
                continue
            p = series_params(DATA_DIR, sname[v.series_uid])
            w.writerow([k, v.orientation, tp_name(v) if v.tp is not None else "-",
                        sname[v.series_uid], p["t"], p["seq"], p["tr"], p["te"],
                        p["fa"], p["nex"], f"{p['mtx']}x{p['mtx']}",
                        round(p["px"], 6), p["n"], round(scale[v.series_uid], 4)])

    # --- 스케일 보정 검증 (DCE 계열만: 프로토콜이 같아야 배경 잡음이 불변량이 된다)
    check_rows, check_plot = [], {}
    for ori in sorted({v.orientation for v in dce}):
        vs = sorted([v for v in dce if v.orientation == ori], key=lambda v: v.sort_key())
        if len({v.data.shape for v in vs}) > 1:
            print(f"  [경고] {ori}: 시점 간 shape 불일치 -> 배경 검증 생략", flush=True)
            continue
        mean = np.mean([v.data for v in vs], axis=0)
        air = mean < np.percentile(mean, AIR_PCT)
        raw, cor = [], []
        for v in vs:
            vr = next(r for r in vols_raw if r.series_uid == v.series_uid)
            s = scale[v.series_uid]
            raw.append(float(np.median(vr.data[air])))
            cor.append(float(np.median(v.data[air])))
            check_rows.append([ori, tp_name(v), sname[v.series_uid], round(s, 4),
                               round(raw[-1], 1), round(cor[-1], 3)])
        check_plot[ori] = ([tp_name(v) for v in vs], raw, cor)
        dev = max(abs(x / cor[0] * 100 - 100) for x in cor)
        devr = max(abs(x / raw[0] * 100 - 100) for x in raw)
        print(f"  {ori}: 배경 잡음 편차 보정 전 {devr:.1f}% -> 보정 후 {dev:.1f}%", flush=True)

    with open(os.path.join(OUT_DIR, "scale_factors.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["orientation", "timepoint", "series_folder", "scale_tag_00E1_1001",
                    "bg_median_raw", "bg_median_corrected"])
        w.writerows(check_rows)

    if check_plot:
        fig, axes = plt.subplots(1, len(check_plot), figsize=(5.5 * len(check_plot), 4),
                                 squeeze=False)
        for ax, ori in zip(axes[0], sorted(check_plot)):
            labs, raw, cor = check_plot[ori]
            x = np.arange(len(labs))
            ax.plot(x, np.array(raw) / raw[0] * 100, "o-", label="보정 전 (raw)")
            ax.plot(x, np.array(cor) / cor[0] * 100, "s-", label="보정 후 (/태그)")
            ax.axhline(100, color="k", lw=0.6, ls=":")
            ax.set_xticks(x); ax.set_xticklabels(labs)
            ax.set_ylabel("배경 잡음 중앙값 (PRE=100%)"); ax.set_title(ori)
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
        fig.suptitle(f"{ANIMAL} 스케일 보정 검증: 불변이어야 할 공기 배경 잡음", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, "scale_check_background.png"), dpi=130)
        plt.close(fig)

    # --- Slab 구성 + 정합
    report, slabs = [], []
    for ori in sorted({v.orientation for v in dce}):
        vs = sorted([v for v in dce if v.orientation == ori], key=lambda v: v.sort_key())
        if vs[0].tp != 0:
            print(f"  [경고] {ori}: PRE 없음 -> 건너뜀", flush=True)
            continue
        pre_v, posts = vs[0], vs[1:]
        if not posts:
            print(f"  [경고] {ori}: POST 없음 -> 건너뜀", flush=True)
            continue
        merged, dks = [pre_v], []
        for pv in posts:
            mv, (dk, dj, di) = resample_to(pre_v, pv, ippmin)
            merged.append(mv); dks.append(dk)
            if abs(dk) > 1e-3 or abs(dj) > 1e-3 or abs(di) > 1e-3:
                print(f"  {ori} {tp_name(pv)}: PRE 격자로 리샘플 "
                      f"(면외 {dk:+.2f} slice, 면내 {dj:+.2f}/{di:+.2f} px)", flush=True)
        nz = pre_v.n_slices
        dks = [0.0 if abs(k) < 0.01 else k for k in dks]
        lo = max(0, int(np.ceil(max(0.0, *[-k for k in dks]))))
        hi = min(nz - 1, int(np.floor(min(float(nz - 1), *[nz - 1 - k for k in dks]))))
        print(f"  {ori}: 시점 {[tp_name(v) for v in merged]}, "
              f"이동 기준 유효 범위 {lo}~{hi} (총 {nz})", flush=True)
        report.append([ANIMAL, ori, nz, lo, hi,
                       round(max([abs(k) for k in dks] or [0.0]), 3),
                       " ".join(tp_name(v) for v in merged)])
        slabs.append(Slab(volumes=merged, orientation=ori, animal=ANIMAL))

    slabs = [register_slab(s, "inplane", True, verbose=True) for s in slabs]

    # --- 분석 마스크 = 조직 ∩ 공통 지지영역
    # 물리좌표 리샘플과 정합 리샘플 모두 FOV 밖을 0으로 채운다. 그 화소를 빼지 않으면
    # (post=0, pre=신호) 때문에 -100%가 섞여 증강이 실제보다 낮게 나온다.
    MASKS, ZS = {}, {}
    for s in slabs:
        pre = s.pre.data
        tis = pre > float(np.mean(pre[pre > 0]))
        support = np.all(np.stack([v.data for v in s.volumes]) > 0, axis=0)
        m = tis & support
        MASKS[s.orientation] = m
        keep = [z for z in range(s.n_slices)
                if tis[z].sum() > 0 and m[z].sum() >= 0.5 * tis[z].sum()]
        ZS[s.orientation] = keep
        drop = 100.0 * (1.0 - m.sum() / max(int(tis.sum()), 1))
        print(f"  {s.orientation}: 조직 화소의 {drop:.1f}%가 0채움이라 제외, "
              f"분석 슬라이스 {keep[0]}~{keep[-1]} ({len(keep)}/{s.n_slices})", flush=True)
        for row in report:
            if row[1] == s.orientation:
                row += [round(drop, 2), f"{keep[0]}~{keep[-1]}", len(keep)]

    with open(os.path.join(OUT_DIR, "valid_slice_ranges.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rat", "orientation", "n_slices", "shift_lo", "shift_hi",
                    "max_shift_slice", "timepoints",
                    "dropped_tissue_vox_pct", "analysis_slices", "n_analysis_slices"])
        w.writerows(report)

    # --- 보조 시리즈(TR200, FSE T1): 스케일 보정 + 같은 방향 PRE 격자로 리샘플
    AUX = {}
    aux_rows = []
    for kind, v in aux:
        ori = v.orientation
        base = next((s.pre for s in slabs if s.orientation == ori), None)
        note = ""
        arr = v.data
        if base is None:
            note = "같은 방향 DCE 없음"
        elif v.data.shape[1:] == base.data.shape[1:]:
            mv, (dk, dj, di) = resample_to(base, v, ippmin)
            arr = mv.data
            if abs(dk) > 1e-3 or abs(dj) > 1e-3 or abs(di) > 1e-3:
                note = f"PRE 격자로 리샘플 (면외 {dk:+.3f} slice, 면내 {dj:+.2f}/{di:+.2f} px)"
                print(f"  [보조] {sname[v.series_uid]}: {note}", flush=True)
        else:
            note = (f"행렬 {v.data.shape[2]} != DCE {base.data.shape[2]} "
                    f"-> 표시용으로만 확대")
        AUX.setdefault(ori, {})[kind] = (arr, v, note)
        p = series_params(DATA_DIR, sname[v.series_uid])
        bg = float(np.median(arr[arr > 0])) if (arr > 0).any() else float("nan")
        aux_rows.append([kind, ori, sname[v.series_uid], p["seq"], p["tr"], p["te"],
                         p["fa"], p["nex"], f"{p['mtx']}x{p['mtx']}",
                         round(scale[v.series_uid], 4), round(bg, 3), note])

    if aux_rows:
        with open(os.path.join(OUT_DIR, "other_series.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["kind", "orientation", "series_folder", "sequence", "TR_ms",
                        "TE_ms", "flip_deg", "NEX", "matrix", "scale_tag_00E1_1001",
                        "median_nonzero_corrected", "note"])
            w.writerows(aux_rows)

    # --- 스케일 보정 볼륨 저장 (리샘플·정합까지 끝난 상태, 시점 간 절대 비교 가능)
    npz = {f"{s.orientation}_{tp_name(v)}": v.data.astype(np.float32)
           for s in slabs for v in s.volumes}
    npz.update({f"{s.orientation}_analysis_mask": MASKS[s.orientation] for s in slabs})
    npz.update({f"{s.orientation}_spacing": np.array(s.spacing, float) for s in slabs})
    for ori, dd in AUX.items():
        for kind, (arr, v, note) in dd.items():
            npz[f"{ori}_{kind.upper()}"] = arr.astype(np.float32)
    np.savez_compressed(os.path.join(OUT_DIR, "corrected_volumes.npz"), **npz)

    # --- 증강 곡선
    summary = []
    curves = [enhancement_curve(s, MASKS[s.orientation]) for s in slabs]
    with open(os.path.join(OUT_DIR, "enhancement_summary.csv"), "w",
              newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["label", "orientation", "n_frames", "roi_mm3", "signal_by_frame",
                    "enh_pct_by_frame", "peak_pct", "time_to_peak_frame", "washin", "auc"])
        for s, cv in zip(slabs, curves):
            row = [cv["label"], s.orientation, cv["n_frames"], cv["roi_mm3"],
                   " ".join(f"{x:.1f}" for x in cv["signal"]),
                   " ".join(f"{x:+.1f}" for x in cv["enh_pct"]),
                   cv["peak_pct"], cv["time_to_peak_frame"],
                   cv["washin_pct_per_unit"], cv["auc_pct"]]
            w.writerow(row)
            summary.append([key] + row)
            print(f"  [{cv['label']}] enh% = {cv['enh_pct']}  peak={cv['peak_pct']}%", flush=True)
    plot_curves(curves, os.path.join(OUT_DIR, "all_curves.png"))

    # --- 전 슬라이스 추출
    for s in slabs:
        tag = s.label()
        montage_anatomy(s, "pre", os.path.join(OUT_DIR, f"{tag}_preGd_anatomy.png"))
        montage_anatomy(s, "peak", os.path.join(OUT_DIR, f"{tag}_peakGd_anatomy.png"))
        montage_heatmap(s, os.path.join(OUT_DIR, f"{tag}_heatmap.png"))

        d_ind = os.path.join(OUT_DIR, s.orientation, "individual")
        d_hm = os.path.join(OUT_DIR, s.orientation, "heatmap_rows")
        d_sub = os.path.join(OUT_DIR, s.orientation, "subtraction_rows")
        for d in (d_ind, d_hm, d_sub):
            os.makedirs(d, exist_ok=True)
        wl = np.percentile(s.volumes[-1].data, [1, 99])
        for z in range(s.n_slices):
            save_frame_slices(s, z, d_ind, wl=wl)
            heatmap_frames_row(s, z, os.path.join(d_hm, f"{tag}_slice{z:02d}_heatmap_row.png"))
            subtraction_frames_row(s, z, os.path.join(d_sub, f"{tag}_slice{z:02d}_subtraction_row.png"))
        print(f"  {tag}: 전체 {s.n_slices}슬라이스 저장", flush=True)

    # --- PRE vs POST5 차감 / 히트맵
    d_p5 = os.path.join(OUT_DIR, "pre_vs_post5")
    d_4p = os.path.join(d_p5, "4panel")
    d_heat = os.path.join(d_p5, "heatmap")
    for d in (d_p5, d_4p, d_heat):
        os.makedirs(d, exist_ok=True)
    stat_rows = []

    for s in slabs:
        tag = s.label()
        tps = [v.tp for v in s.volumes]
        if 5 not in tps:
            print(f"  [경고] {tag}: POST5 없음(시점 {tps}) -> 차감 생략", flush=True)
            continue
        pre = s.volumes[0].data.astype(float)
        p5 = s.volumes[tps.index(5)].data.astype(float)
        delta = p5 - pre
        tis = MASKS[s.orientation]
        enh = np.where(tis, delta / (pre + 1e-6) * 100.0, np.nan)
        zs = ZS[s.orientation]
        lo, hi = zs[0], zs[-1]
        wl = np.percentile(p5[zs], [1, 99])
        M = float(np.percentile(np.abs(delta[tis]), 99)) or 1.0
        vox_mm2 = abs(s.spacing[0] * s.spacing[1])

        for z in zs:
            mz = tis[z]

            fig, ax = plt.subplots(figsize=(5.2, 5.0))
            im = ax.imshow(delta[z], cmap="seismic", vmin=-M, vmax=M)
            ax.axis("off")
            ax.set_title(f"{tag}  slice {z}   dS = POST5 - PRE(스케일 보정)", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="dS (빨강=증가)")
            fig.savefig(os.path.join(d_p5, f"{tag}_slice{z:02d}_delta.png"),
                        dpi=140, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(5.2, 5.0))
            ax.imshow(p5[z], cmap="gray", vmin=wl[0], vmax=wl[1])
            im = ax.imshow(enh[z], cmap="jet", vmin=0, vmax=HEAT_VMAX, alpha=0.55)
            ax.axis("off")
            ax.set_title(f"{tag}  slice {z}   (POST5-PRE)/PRE", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label="% 증강")
            fig.savefig(os.path.join(d_heat, f"{tag}_slice{z:02d}_heat.png"),
                        dpi=140, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
            ax[0].imshow(pre[z], cmap="gray", vmin=wl[0], vmax=wl[1]); ax[0].set_title("PRE (스케일 보정)")
            ax[1].imshow(p5[z], cmap="gray", vmin=wl[0], vmax=wl[1]); ax[1].set_title("POST5")
            im2 = ax[2].imshow(delta[z], cmap="seismic", vmin=-M, vmax=M)
            ax[2].set_title("dS = POST5 - PRE")
            ax[3].imshow(p5[z], cmap="gray", vmin=wl[0], vmax=wl[1])
            im3 = ax[3].imshow(enh[z], cmap="jet", vmin=0, vmax=HEAT_VMAX, alpha=0.55)
            ax[3].set_title("(POST5-PRE)/PRE  %")
            for a in ax:
                a.axis("off")
            fig.suptitle(f"{tag}  slice {z}  (0-index)", fontsize=12)
            fig.subplots_adjust(right=0.9)
            fig.colorbar(im2, cax=fig.add_axes([0.485, 0.12, 0.008, 0.7]))
            fig.colorbar(im3, cax=fig.add_axes([0.92, 0.12, 0.010, 0.7]), label="% enh")
            fig.savefig(os.path.join(d_4p, f"{tag}_slice{z:02d}_pre_post5.png"),
                        dpi=130, bbox_inches="tight")
            plt.close(fig)

            e = enh[z][mz]
            d_ = delta[z][mz]
            stat_rows.append([
                s.orientation, z, int(mz.sum()),
                round(float(np.mean(pre[z][mz])), 1), round(float(np.mean(p5[z][mz])), 1),
                round(float(np.mean(d_)), 2), round(float(np.percentile(d_, 95)), 2),
                round(float(np.mean(e)), 1), round(float(np.median(e)), 1),
                round(float(np.percentile(e, 95)), 1), round(float(np.max(e)), 1),
                int((e > 15).sum()), int((e > 25).sum()), int((e > 50).sum()),
                round(float((e > 25).sum()) * vox_mm2, 2)])

        n = len(zs)
        cols = min(7, n); rows_ = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows_, cols, figsize=(2.6 * cols, 2.8 * rows_), squeeze=False)
        axes = axes.ravel()
        im = None
        for k, z in enumerate(zs):
            im = axes[k].imshow(delta[z], cmap="seismic", vmin=-M, vmax=M)
            axes[k].set_title(f"slice {z}", fontsize=8)
        for a in axes:
            a.axis("off")
        fig.suptitle(f"{tag}  dS = POST5 - PRE  (분석 슬라이스 {lo}~{hi}, 빨강=증가)", fontsize=12)
        fig.subplots_adjust(right=0.9)
        fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.010, 0.7]), label="dS")
        fig.savefig(os.path.join(d_p5, f"{tag}_montage_delta_post5_pre.png"),
                    dpi=130, bbox_inches="tight")
        plt.close(fig)

        for vmax, suffix in ((HEAT_VMAX, ""), (HEAT_VMAX_BRAIN, f"_vmax{HEAT_VMAX_BRAIN}")):
            fig, axes = plt.subplots(rows_, cols, figsize=(2.6 * cols, 2.8 * rows_), squeeze=False)
            axes = axes.ravel()
            im = None
            for k, z in enumerate(zs):
                axes[k].imshow(p5[z], cmap="gray", vmin=wl[0], vmax=wl[1])
                im = axes[k].imshow(enh[z], cmap="jet", vmin=0, vmax=vmax, alpha=0.55)
                axes[k].set_title(f"slice {z}", fontsize=8)
            for a in axes:
                a.axis("off")
            fig.suptitle(f"{tag}  (POST5-PRE)/PRE  (분석 슬라이스 {lo}~{hi}, 상한 {vmax}%)", fontsize=12)
            fig.subplots_adjust(right=0.9)
            fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.010, 0.7]), label="% 증강")
            fig.savefig(os.path.join(d_p5, f"{tag}_montage_heat_post5{suffix}.png"),
                        dpi=130, bbox_inches="tight")
            plt.close(fig)

        print(f"  {tag}: 조직 평균 PRE={pre[tis].mean():.1f} POST5={p5[tis].mean():.1f} "
              f"dS={delta[tis].mean():+.1f} ({delta[tis].mean() / pre[tis].mean() * 100:+.1f}%)",
              flush=True)

    if stat_rows:
        with open(os.path.join(d_p5, "pre_vs_post5_slice_stats.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["orientation", "slice_0idx", "tissue_vox",
                        "pre_mean", "post5_mean", "dS_mean", "dS_p95",
                        "enh_mean_pct", "enh_median_pct", "enh_p95_pct", "enh_max_pct",
                        "vox_enh_gt15", "vox_enh_gt25", "vox_enh_gt50", "area_gt25_mm2"])
            w.writerows(stat_rows)

    # --- 보조 시리즈 출력: 몽타주 + 슬라이스별 + DCE와 나란히 보기
    d_ot = os.path.join(OUT_DIR, "other_series")
    os.makedirs(d_ot, exist_ok=True)
    KNAME = {"tr200": "GRE TR200/FA60 (후기)", "fse": "FSE T1 (Gd 후, 대응 PRE 없음)"}
    for s in slabs:
        ori, tag = s.orientation, s.label()
        dd = AUX.get(ori, {})
        if not dd:
            continue
        zs = ZS[ori]
        pre = s.volumes[0].data.astype(float)
        tps = [v.tp for v in s.volumes]
        p5 = s.volumes[tps.index(5)].data.astype(float) if 5 in tps else s.volumes[-1].data.astype(float)
        wl = np.percentile(p5[zs], [1, 99])

        for kind, (arr, v, note) in dd.items():
            a = arr.astype(float)
            wla = np.percentile(a[a > 0], [1, 99]) if (a > 0).any() else (0, 1)
            n = a.shape[0]
            cols = min(7, n); rows_ = int(np.ceil(n / cols))
            fig, axes = plt.subplots(rows_, cols, figsize=(2.6 * cols, 2.8 * rows_), squeeze=False)
            axes = axes.ravel()
            for k in range(n):
                axes[k].imshow(a[k], cmap="gray", vmin=wla[0], vmax=wla[1])
                axes[k].set_title(f"slice {k}", fontsize=8)
            for ax_ in axes:
                ax_.axis("off")
            fig.suptitle(f"{tag}  {KNAME[kind]}  (스케일 보정){'  ' + note if note else ''}",
                         fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(d_ot, f"{tag}_{kind}_montage.png"),
                        dpi=130, bbox_inches="tight")
            plt.close(fig)

            d_k = os.path.join(d_ot, f"{tag}_{kind}")
            os.makedirs(d_k, exist_ok=True)
            for k in range(n):
                fig, ax_ = plt.subplots(figsize=(5.0, 5.0))
                ax_.imshow(a[k], cmap="gray", vmin=wla[0], vmax=wla[1])
                ax_.axis("off")
                ax_.set_title(f"{tag}  {KNAME[kind]}  slice {k}", fontsize=10)
                fig.savefig(os.path.join(d_k, f"{tag}_{kind}_slice{k:02d}.png"),
                            dpi=140, bbox_inches="tight")
                plt.close(fig)

        # 나란히 보기: PRE / POST5 / TR200 / FSE
        order = [("PRE", pre, wl), ("POST5", p5, wl)]
        for kind in ("tr200", "fse"):
            if kind in dd:
                a = zoom_to(dd[kind][0].astype(float), pre.shape[1], pre.shape[2])
                wla = np.percentile(a[a > 0], [1, 99]) if (a > 0).any() else (0, 1)
                order.append((KNAME[kind].split(" (")[0], a, wla))
        d_cmp = os.path.join(d_ot, f"{tag}_compare")
        os.makedirs(d_cmp, exist_ok=True)
        for z in zs:
            fig, axs = plt.subplots(1, len(order), figsize=(4.0 * len(order), 4.4))
            for ax_, (nm, a, w_) in zip(np.atleast_1d(axs), order):
                if z < a.shape[0]:
                    ax_.imshow(a[z], cmap="gray", vmin=w_[0], vmax=w_[1])
                ax_.set_title(nm, fontsize=10); ax_.axis("off")
            fig.suptitle(f"{tag}  slice {z} (0-index) — 모두 스케일 보정. "
                         f"TR200/FSE는 프로토콜이 달라 밝기를 서로 비교하지 않는다.",
                         fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(d_cmp, f"{tag}_slice{z:02d}_compare.png"),
                        dpi=130, bbox_inches="tight")
            plt.close(fig)
        print(f"  {tag}: 보조 시리즈 {list(dd)} 저장", flush=True)

    return summary


# ------------------------------------------------------------ 뇌 확대 비교 그림
def focal_figs(key, cfg):
    """저장된 corrected_volumes.npz만 읽어 뇌 중심부 확대 패널을 만든다.

    PRE / POST5 / FSE T1 / 증강% 를 같은 크롭으로 나란히 놓아, 증강이 Gd 누출인지
    PRE부터 있던 구조인지 눈으로 구분할 수 있게 한다. FSE는 표시용으로만 격자를 키운다.
    """
    OUT_DIR, ANIMAL = cfg["out"], cfg["label"]
    f = os.path.join(OUT_DIR, "corrected_volumes.npz")
    if not os.path.exists(f):
        print(f"  [경고] {f} 없음 -> 건너뜀", flush=True)
        return
    d = np.load(f)
    d_out = os.path.join(OUT_DIR, "brain_zoom")
    os.makedirs(d_out, exist_ok=True)
    for ori in ("axial", "coronal"):
        if f"{ori}_PRE" not in d.files:
            continue
        pre = d[f"{ori}_PRE"].astype(float)
        p5 = d[f"{ori}_POST5"].astype(float)
        m = d[f"{ori}_analysis_mask"]
        enh = np.where(m, (p5 - pre) / (pre + 1e-6) * 100.0, np.nan)
        panels = [("PRE", pre), ("POST5", p5)]
        if f"{ori}_FSE" in d.files:
            fse = d[f"{ori}_FSE"].astype(float)
            panels.append(("FSE T1", zoom_to(fse, pre.shape[1], pre.shape[2])))
        idx = np.argwhere(m.any(0))
        r0, c0 = idx.min(0); r1, c1 = idx.max(0) + 1
        dr, dc = int((r1 - r0) * 0.22), int((c1 - c0) * 0.22)
        r0, r1, c0, c1 = r0 + dr, r1 - dr, c0 + dc, c1 - dc
        wl = np.percentile(pre[pre > 0], [2, 99.5])
        for z in range(pre.shape[0]):
            fig, ax = plt.subplots(1, len(panels) + 1,
                                   figsize=(4.0 * (len(panels) + 1), 4.6))
            for a, (nm, vol) in zip(ax, panels):
                w = wl if nm != "FSE T1" else np.percentile(vol[vol > 0], [2, 99.5])
                a.imshow(vol[z, r0:r1, c0:c1], cmap="gray", vmin=w[0], vmax=w[1])
                a.set_title(nm, fontsize=10)
            im = ax[-1].imshow(enh[z, r0:r1, c0:c1], cmap="jet", vmin=0, vmax=60)
            ax[-1].set_title("(POST5-PRE)/PRE %", fontsize=10)
            for a in ax:
                a.set_xticks(np.arange(0, c1 - c0, 20)); a.set_yticks(np.arange(0, r1 - r0, 20))
                a.set_xticklabels(np.arange(c0, c1, 20), fontsize=6)
                a.set_yticklabels(np.arange(r0, r1, 20), fontsize=6)
                a.grid(alpha=0.25, color="w", lw=0.4)
            fig.colorbar(im, ax=ax[-1], fraction=0.046, pad=0.03, label="% 증강")
            fig.suptitle(f"{ANIMAL}_{ori}  slice {z} (0-index)  "
                         f"crop row {r0}~{r1} col {c0}~{c1}", fontsize=11)
            fig.tight_layout()
            fig.savefig(os.path.join(d_out, f"{ANIMAL}_{ori}_slice{z:02d}_zoom.png"),
                        dpi=120, bbox_inches="tight")
            plt.close(fig)
        print(f"  {ANIMAL}_{ori}: 뇌 확대 {pre.shape[0]}장 저장", flush=True)


# ------------------------------------------------------------ 좌우 대칭 비교
def mirror_axis(vol, mask):
    """열 방향 대칭축(중앙선) 위치를 PRE 영상 자체에서 찾는다.

    두 방향 모두 IOP의 행 방향 r = (-1,0,0)이라 **열 축이 곧 좌우 축**이다.
    거울상 인덱스 c' = 2*c0 - c 가 되도록 flip 후 평행이동하고, 마스크 안에서
    원본과의 상관이 최대가 되는 c0을 0.25화소 간격으로 찾는다.
    반환: (c0, 최대 상관). 상관이 낮으면 머리가 기울어 좌우 비교가 부정확하다는 뜻이다.
    """
    N = vol.shape[2]
    f = np.flip(vol, axis=2)
    fm = np.flip(mask, axis=2)
    # 마스크 열 중심 ±20화소만 본다. 축을 멀리 옮기면 겹치는 영역이 줄어
    # 상관이 허수로 올라가므로, 겹침이 마스크의 75% 미만이면 버린다.
    cen = float(np.argwhere(mask)[:, 2].mean())
    need = 0.75 * mask.sum()

    def score(c0):
        g = ndimage.shift(f, (0, 0, 2 * c0 - (N - 1)), order=1, mode="constant", cval=0.0)
        gm = ndimage.shift(fm.astype(np.float32), (0, 0, 2 * c0 - (N - 1)),
                           order=0, mode="constant", cval=0.0) > 0.5
        k = mask & gm
        if k.sum() < need:
            return None
        return float(np.corrcoef(vol[k], g[k])[0, 1])

    best = (cen, -1.0)
    for c0 in np.arange(cen - 20, cen + 20 + 1e-9, 1.0):
        r = score(c0)
        if r is not None and r > best[1]:
            best = (float(c0), r)
    for c0 in np.arange(best[0] - 1.0, best[0] + 1.0 + 1e-9, 0.25):
        r = score(c0)
        if r is not None and r > best[1]:
            best = (float(c0), r)
    return best


def mirror_asym(key, cfg):
    """좌우 거울상 차이로 편측 증강을 본다(대측 반구가 곧 내부 대조군).

    FUS-BBBO는 보통 한쪽만 연다. 두피·측두근은 좌우가 대체로 대칭이므로 거울상 차이를
    보면 뇌 분할 없이도 편측 성분만 남길 수 있다. 남는 한계는 그대로 있다:
    머리가 기울면 정렬 오차가 차이로 나타나고, 원래 좌우 비대칭인 구조(혈관 등)도 남는다.
    """
    OUT_DIR, ANIMAL = cfg["out"], cfg["label"]
    f = os.path.join(OUT_DIR, "corrected_volumes.npz")
    if not os.path.exists(f):
        return []
    d = np.load(f)
    d_out = os.path.join(OUT_DIR, "mirror_asym")
    os.makedirs(d_out, exist_ok=True)
    rows = []
    for ori in ("axial", "coronal"):
        if f"{ori}_PRE" not in d.files:
            continue
        pre = d[f"{ori}_PRE"].astype(float)
        p5 = d[f"{ori}_POST5"].astype(float)
        m = d[f"{ori}_analysis_mask"]
        sp = d[f"{ori}_spacing"]
        c0, r = mirror_axis(pre, m)
        N = pre.shape[2]
        enh = np.where(m, (p5 - pre) / (pre + 1e-6) * 100.0, 0.0)
        sh = 2 * c0 - (N - 1)
        emir = ndimage.shift(np.flip(enh, axis=2), (0, 0, sh), order=1,
                             mode="constant", cval=0.0)
        mmir = ndimage.shift(np.flip(m, axis=2).astype(np.float32), (0, 0, sh),
                             order=0, mode="constant", cval=0.0) > 0.5
        both = m & mmir
        dasym = np.where(both, enh - emir, np.nan)
        print(f"  {ANIMAL}_{ori}: 대칭축 col={c0:.2f} (좌우 상관 {r:.3f}), "
              f"양쪽 유효 화소 {int(both.sum())}", flush=True)

        M = float(np.nanpercentile(np.abs(dasym), 99)) or 1.0
        wl = np.percentile(p5[p5 > 0], [2, 99])
        n = pre.shape[0]
        cols = min(7, n); rr = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rr, cols, figsize=(2.6 * cols, 2.8 * rr), squeeze=False)
        axes = axes.ravel(); im = None
        for z in range(n):
            axes[z].imshow(p5[z], cmap="gray", vmin=wl[0], vmax=wl[1])
            im = axes[z].imshow(dasym[z], cmap="seismic", vmin=-M, vmax=M, alpha=0.6)
            axes[z].set_title(f"slice {z}", fontsize=8)
        for a in axes:
            a.axis("off")
        fig.suptitle(f"{ANIMAL}_{ori}  증강% 좌우 거울상 차이 (자기 - 반대편), "
                     f"대칭축 col={c0:.1f}", fontsize=12)
        fig.subplots_adjust(right=0.9)
        fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.010, 0.7]), label="%p")
        fig.savefig(os.path.join(d_out, f"{ANIMAL}_{ori}_mirror_asym_montage.png"),
                    dpi=130, bbox_inches="tight")
        plt.close(fig)

        vox = abs(sp[0] * sp[1])
        for z in range(n):
            a = dasym[z][both[z]]
            if a.size < 100:
                continue
            sm = ndimage.gaussian_filter(np.nan_to_num(dasym[z]), 2.0)
            sm = np.where(both[z], sm, np.nan)
            j, i = np.unravel_index(np.nanargmax(sm), sm.shape)
            rows.append([ori, z, round(c0, 2), round(r, 3), int(both[z].sum()),
                         round(float(np.median(a)), 2), round(float(np.percentile(a, 95)), 2),
                         round(float(np.nanmax(sm)), 2), int(j), int(i),
                         int((a > 10).sum()), round(float((a > 10).sum()) * vox, 2)])
    if rows:
        with open(os.path.join(d_out, "mirror_asym_slice_stats.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["orientation", "slice_0idx", "mirror_axis_col", "lr_corr",
                        "paired_vox", "asym_median_pp", "asym_p95_pp",
                        "asym_max_smoothed_pp", "peak_row", "peak_col",
                        "vox_asym_gt10pp", "area_gt10pp_mm2"])
            w.writerows(rows)
    return rows


# ------------------------------------------------------------ 편측 병소 정량
FOCUS_ASYM_PP = 20      # 편측 후보 문턱 (%p)
FOCUS_MIN_VOX = 40      # 최소 화소수
FOCUS_ERODE = 21        # 두피/근육 제외용 침식 커널(화소)


def lateral_focus(key, cfg):
    """좌우 비대칭이 큰 덩어리를 찾아 대측 거울상과 정량 비교한다.

    대측 거울상 ROI가 대조군이다. 순 누출 = 병소 증강% - 대측 증강%.
    (mri_dce의 bbbo_metrics와 같은 정의. 15%p 이상이면 양성으로 본다.)

    한계를 분명히 해 둔다. 여기에는 해부학적 뇌 분할이 없다. 두피·근육을 빼려고
    조직 마스크를 21화소 침식했을 뿐이라 목덜미 근육이나 영상 가장자리 덩어리가
    후보에 남는다. 중심 좌표를 함께 적으니 그림과 대조해서 걸러야 한다.
    또 좌우 대칭축은 평행이동만 맞춘다. 머리가 기울면 그만큼 오차가 남는다.
    """
    OUT_DIR, ANIMAL = cfg["out"], cfg["label"]
    f = os.path.join(OUT_DIR, "corrected_volumes.npz")
    if not os.path.exists(f):
        return []
    d = np.load(f)
    d_out = os.path.join(OUT_DIR, "lateral_focus")
    os.makedirs(d_out, exist_ok=True)
    rows = []
    for ori in ("axial", "coronal"):
        if f"{ori}_PRE" not in d.files:
            continue
        tps = ["PRE"] + [f"POST{i}" for i in range(1, 9) if f"{ori}_POST{i}" in d.files]
        vols = [d[f"{ori}_{t}"].astype(float) for t in tps]
        pre, p5 = vols[0], vols[-1]
        m = d[f"{ori}_analysis_mask"]
        sp = d[f"{ori}_spacing"]
        N = pre.shape[2]
        c0, lr = mirror_axis(pre, m)
        sh = 2 * c0 - (N - 1)
        enh = np.where(m, (p5 - pre) / (pre + 1e-6) * 100.0, 0.0)
        emir = ndimage.shift(np.flip(enh, axis=2), (0, 0, sh), order=1,
                             mode="constant", cval=0.0)
        mmir = ndimage.shift(np.flip(m, axis=2).astype(np.float32), (0, 0, sh),
                             order=0, mode="constant", cval=0.0) > 0.5
        asym = ndimage.gaussian_filter(np.where(m & mmir, enh - emir, 0.0), (0, 1.5, 1.5))
        inner = ndimage.binary_erosion(m & mmir, np.ones((1, FOCUS_ERODE, FOCUS_ERODE)))
        cand = ndimage.binary_opening(inner & (asym > FOCUS_ASYM_PP), np.ones((1, 3, 3)))
        lab, n = ndimage.label(cand)
        if n == 0:
            print(f"  {ANIMAL}_{ori}: {FOCUS_ASYM_PP}%p 넘는 편측 덩어리 없음", flush=True)
            continue
        sizes = ndimage.sum(cand, lab, range(1, n + 1))
        fse = zoom_to(d[f"{ori}_FSE"].astype(float), pre.shape[1], pre.shape[2])             if f"{ori}_FSE" in d.files else None
        vox = abs(sp[0] * sp[1] * sp[2])
        for rank, k in enumerate(np.argsort(sizes)[::-1][:3], 1):
            if sizes[k] < FOCUS_MIN_VOX:
                break
            sel = lab == (k + 1)
            mir = np.zeros_like(sel)
            for z_, r_, c_ in np.argwhere(sel):
                cb = int(round(2 * c0 - c_))
                if 0 <= cb < N:
                    mir[z_, r_, cb] = True
            zc, rc, cc = np.argwhere(sel).mean(0)
            sig = [float(v[sel].mean()) for v in vols]
            sgm = [float(v[mir].mean()) for v in vols]
            e = [(x - sig[0]) / sig[0] * 100.0 for x in sig]
            em = [(x - sgm[0]) / sgm[0] * 100.0 for x in sgm]
            net = e[-1] - em[-1]
            zs = sorted({int(z_) for z_ in np.argwhere(sel)[:, 0]})
            fr = (float(fse[sel].mean() / fse[mir].mean())
                  if fse is not None and fse[mir].mean() > 0 else float("nan"))
            # 절대 신호차도 같이 남긴다. 병소 PRE가 낮으면 %증강은 분모 때문에
            # 부풀려지므로, 분모 없는 dS로 교차 확인할 수 있어야 한다.
            dsf, dsm = sig[-1] - sig[0], sgm[-1] - sgm[0]
            rows.append([ori, rank, int(sizes[k]), round(sizes[k] * vox, 2),
                         round(c0, 2), round(lr, 3),
                         round(zc, 1), int(round(rc)), int(round(cc)),
                         " ".join(str(z_) for z_ in zs),
                         round(float(asym[sel].mean()), 1), round(float(asym[sel].max()), 1),
                         " ".join(f"{x:+.1f}" for x in e),
                         " ".join(f"{x:+.1f}" for x in em),
                         round(net, 1), "양성" if net >= 15 else "음성",
                         round(em[-1], 1),
                         round(sig[0], 1), round(sgm[0], 1), round(sig[0] / sgm[0], 3),
                         round(dsf, 1), round(dsm, 1), round(dsf - dsm, 1), round(fr, 3)])
            print(f"  {ANIMAL}_{ori} #{rank}: {int(sizes[k])}화소 {sizes[k]*vox:.2f}mm3 "
                  f"중심(z {zc:.1f}, row {rc:.0f}, col {cc:.0f}) 슬라이스 {zs}  "
                  f"순 누출 {net:+.1f}%p  PRE비 {sig[0]/sgm[0]:.2f}", flush=True)

            # 그림: 해당 슬라이스마다 PRE / POST5 / FSE / 비대칭 + ROI 윤곽
            wl = np.percentile(pre[pre > 0], [2, 99.5])
            for z_ in zs:
                pan = [("PRE", pre, wl), ("POST5", p5, wl)]
                if fse is not None:
                    pan.append(("FSE T1", fse, np.percentile(fse[fse > 0], [2, 99.5])))
                fig, ax = plt.subplots(1, len(pan) + 1, figsize=(4.0 * (len(pan) + 1), 4.5))
                for a, (nm, vv, ww) in zip(ax, pan):
                    a.imshow(vv[z_], cmap="gray", vmin=ww[0], vmax=ww[1])
                    a.contour(sel[z_], levels=[0.5], colors="r", linewidths=0.9)
                    a.contour(mir[z_], levels=[0.5], colors="c", linewidths=0.9)
                    a.set_title(nm, fontsize=10); a.axis("off")
                im = ax[-1].imshow(asym[z_], cmap="seismic", vmin=-60, vmax=60)
                ax[-1].contour(sel[z_], levels=[0.5], colors="k", linewidths=0.9)
                ax[-1].set_title("좌우 비대칭 %p", fontsize=10); ax[-1].axis("off")
                fig.colorbar(im, ax=ax[-1], fraction=0.046, pad=0.03, label="%p")
                fig.suptitle(f"{ANIMAL}_{ori} 병소#{rank} slice {z_} — 빨강=병소 ROI, "
                             f"하늘=대측 거울 ROI, 순 누출 {net:+.1f}%p", fontsize=11)
                fig.tight_layout()
                fig.savefig(os.path.join(d_out,
                            f"{ANIMAL}_{ori}_focus{rank}_slice{z_:02d}.png"),
                            dpi=125, bbox_inches="tight")
                plt.close(fig)

            fig, a = plt.subplots(figsize=(5.6, 4.2))
            x = np.arange(len(tps))
            a.plot(x, e, "o-", color="crimson", label="병소")
            a.plot(x, em, "s-", color="steelblue", label="대측 거울상")
            a.axhline(0, color="k", lw=0.6, ls=":")
            a.set_xticks(x); a.set_xticklabels(tps)
            a.set_ylabel("증강 % (PRE 기준)")
            a.set_title(f"{ANIMAL}_{ori} 병소#{rank}  순 누출 {net:+.1f}%p", fontsize=11)
            a.legend(); a.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(d_out, f"{ANIMAL}_{ori}_focus{rank}_curve.png"), dpi=130)
            plt.close(fig)
    if rows:
        with open(os.path.join(d_out, "lateral_focus.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            # mirror_enh_last_pct 가 25% 넘으면 ROI가 뇌가 아니라 두피/근육이다.
            # 정상 뇌실질 대측은 이 데이터에서 10% 아래로 나온다.
            w.writerow(["orientation", "rank", "voxels", "volume_mm3",
                        "mirror_axis_col", "lr_corr", "center_z", "center_row",
                        "center_col", "slices", "asym_mean_pp", "asym_max_pp",
                        "focus_enh_pct_by_frame", "mirror_enh_pct_by_frame",
                        "net_leakage_pp_lastPOST", "verdict_15pp",
                        "mirror_enh_last_pct", "PRE_focus", "PRE_mirror",
                        "PRE_focus_over_mirror", "dS_focus", "dS_mirror", "dS_diff",
                        "FSE_focus_over_mirror"])
            w.writerows(rows)
    return rows


# ------------------------------------------------------------ 직교 방향 교차 확인
def _ori_geom(rat_dir):
    """방향별 (첫 슬라이스 IPP, r, c, n, 화소크기, 슬라이스 간격)을 DICOM에서 읽는다.

    PRE 시리즈 기준. 두 방향은 같은 환자 좌표계를 쓰므로, 한쪽에서 찾은 병소 중심을
    다른 쪽 격자 인덱스로 옮길 수 있다.
    """
    out = {}
    for folder in sorted(os.listdir(rat_dir)):
        d = os.path.join(rat_dir, folder)
        if not os.path.isdir(d) or classify(folder) != ("dce", 0):
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
        sz = float(getattr(ds[0], "SpacingBetweenSlices", None)
                   or ds[0].SliceThickness)
        out[classify_orientation(iop)] = (ipp, r, c, n, ps, sz)
    return out


def cross_orientation_check(key, cfg, rows):
    """한 방향에서 찾은 병소를 직교 방향의 같은 물리 좌표에서 다시 본다.

    두 번 따로 찍은 영상에서 같은 자리에 나오면 정합·잡음 탓으로 보기 어렵다.
    비교 기준은 그 슬라이스 뇌 안쪽 비대칭 분포의 p95/p99다.
    """
    if not rows:
        return
    OUT_DIR, DATA_DIR, ANIMAL = cfg["out"], cfg["data"], cfg["label"]
    d = np.load(os.path.join(OUT_DIR, "corrected_volumes.npz"))
    G = _ori_geom(DATA_DIR)
    A = {}
    for ori in ("axial", "coronal"):
        if f"{ori}_PRE" not in d.files or ori not in G:
            continue
        pre = d[f"{ori}_PRE"].astype(float)
        p5 = d[f"{ori}_POST5"].astype(float)
        m = d[f"{ori}_analysis_mask"]
        N = pre.shape[2]
        c0, _ = mirror_axis(pre, m)
        sh = 2 * c0 - (N - 1)
        enh = np.where(m, (p5 - pre) / (pre + 1e-6) * 100.0, 0.0)
        emir = ndimage.shift(np.flip(enh, axis=2), (0, 0, sh), order=1,
                             mode="constant", cval=0.0)
        mmir = ndimage.shift(np.flip(m, axis=2).astype(np.float32), (0, 0, sh),
                             order=0, mode="constant", cval=0.0) > 0.5
        A[ori] = (ndimage.gaussian_filter(np.where(m & mmir, enh - emir, 0.0), (0, 1.5, 1.5)),
                  ndimage.binary_erosion(m & mmir, np.ones((1, FOCUS_ERODE, FOCUS_ERODE))))
    out = []
    for x in rows:
        src = x[0]
        dst = "coronal" if src == "axial" else "axial"
        if src not in G or dst not in A:
            continue
        ipp, r, c, n, ps, sz = G[src]
        P = ipp + x[8] * ps * r + x[7] * ps * c + x[6] * sz * n
        ipd, rd, cd, nd, psd, szd = G[dst]
        dd = P - ipd
        zi = int(round(float(np.dot(dd, nd)) / szd))
        ri = int(round(float(np.dot(dd, cd)) / psd))
        ci = int(round(float(np.dot(dd, rd)) / psd))
        asym, inner = A[dst]
        if not (0 <= zi < asym.shape[0] and 0 <= ri < asym.shape[1] and 0 <= ci < asym.shape[2]):
            out.append([ANIMAL, src, x[1], dst, zi, ri, ci, "", "", "", "", "격자 밖"])
            continue
        patch = asym[zi, max(0, ri - 4):ri + 5, max(0, ci - 4):ci + 5]
        v = asym[zi][inner[zi]]
        med = float(np.median(patch))
        p95 = float(np.percentile(v, 95)) if v.size else float("nan")
        p99 = float(np.percentile(v, 99)) if v.size else float("nan")
        out.append([ANIMAL, src, x[1], dst, zi, ri, ci,
                    round(med, 1), round(float(patch.max()), 1),
                    round(p95, 1), round(p99, 1),
                    "재현" if med >= p99 else ("경계" if med >= p95 else "미재현")])
    if out:
        with open(os.path.join(OUT_DIR, "lateral_focus", "crosscheck.csv"), "w",
                  newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["animal", "src_orientation", "rank", "dst_orientation",
                        "dst_z", "dst_row", "dst_col", "dst_asym_median_pp",
                        "dst_asym_max_pp", "dst_slice_p95_pp", "dst_slice_p99_pp",
                        "verdict"])
            w.writerows(out)
        for o in out:
            print(f"  {ANIMAL} {o[1]}#{o[2]} -> {o[3]} z={o[4]} row={o[5]} col={o[6]}: "
                  f"비대칭 {o[7]}%p (그 슬라이스 p95 {o[9]}, p99 {o[10]}) -> {o[-1]}", flush=True)


# ------------------------------------------------------------ 실행
argv = sys.argv[1:]
FIGS_ONLY = "figs" in argv          # 저장된 npz로 brain_zoom 그림만 다시 만든다
keys = [a for a in argv if a != "figs"] or list(SETS)
bad = [k for k in keys if k not in SETS]
if bad:
    sys.exit(f"알 수 없는 키: {bad}\n사용 가능: {list(SETS)}")

if FIGS_ONLY:
    for k in keys:
        print(f"########## RAT {k} (brain_zoom만) ##########", flush=True)
        focal_figs(k, SETS[k])
        mirror_asym(k, SETS[k])
        cross_orientation_check(k, SETS[k], lateral_focus(k, SETS[k]))
    sys.exit(0)

allsum = []
for k in keys:
    allsum += run_one(k, SETS[k])
    focal_figs(k, SETS[k])
    mirror_asym(k, SETS[k])
    cross_orientation_check(k, SETS[k], lateral_focus(k, SETS[k]))

with open(os.path.join(OUT_ROOT, "enhancement_summary_all.csv"), "w",
          newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["rat", "label", "orientation", "n_frames", "roi_mm3",
                "signal_by_frame", "enh_pct_by_frame", "peak_pct",
                "time_to_peak_frame", "washin", "auc"])
    w.writerows(allsum)

print("\n\n===== 전체 요약 =====", flush=True)
print("%-6s %-9s %6s  %s" % ("rat", "ori", "peak%", "enh% by frame"))
for r in allsum:
    print("%-6s %-9s %6s  %s" % (r[0], r[2], r[7], r[6]))
print("\n완료", flush=True)
