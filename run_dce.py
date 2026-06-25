# -*- coding: utf-8 -*-
"""260625 BBB RAT — 3번 랫만 전 슬라이스 추출 + 표준 배치 분석 (출력 폴더명 RAT_3).

데이터: 260625_BBB_RAT/260625_BBB_RAT/3/{PRE,Post1~5}_GRE_BBB_RAT_{AX,COR}.dcm(폴더, 슬라이스별 파일).
1·2번은 요청에 따라 제외(pre+post1만 존재).
전처리(이전 세션들과 동일): IPP/IOP 물리좌표로 post를 pre 격자에 리샘플 → Slab 수동 구성 →
rigid 정합(guard). 슬라이스 번호는 0-index (DICOM slice N+1).

실행: python _run_all_slices.py
"""
import os
import sys
import csv
from dataclasses import replace
import numpy as np
import pydicom
from scipy import ndimage

CODE_DIR = r"C:\Users\user\Desktop\대학원\6. 원자력의학원\MRI 분석 코드"
BASE = r"C:\Users\user\Desktop\대학원\6. 원자력의학원\26-06-25\260625_BBB_RAT\260625_BBB_RAT"
DAY = "0625"
TAG_DAY = DAY
DATA_DIR = BASE
OUT_DIR = r"C:\Users\user\Desktop\대학원\6. 원자력의학원\26-06-25\2. 분석 결과"

sys.path.insert(0, CODE_DIR)
from mri_dce import (_load_volumes, Slab, register_slab, enhancement_curve,
                     plot_curves, save_frame_slices, heatmap_frames_row,
                     subtraction_frames_row, montage_anatomy, montage_heatmap)


def series_min_ipp(rat_dir):
    """시리즈별 최소-투영 슬라이스의 IPP. (_load_volumes는 proj 오름차순으로 쌓으므로
    배열 z=0 슬라이스의 물리 위치 = 이 값.)"""
    out = {}
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
            if uid not in out or p < out[uid][0]:
                out[uid] = (p, ipp)
    return {uid: ipp for uid, (p, ipp) in out.items()}


def merge_to_pre_grid(pre_v, post_v, ippmin):
    """post 볼륨을 pre 격자로 리샘플(물리좌표 평행이동만; IOP·spacing 동일 전제).
    반환: (병합 Volume, 면외 슬라이스 이동량 dk)."""
    r = np.array(pre_v.iop[:3], float)
    c = np.array(pre_v.iop[3:], float)
    n = np.cross(r, c)
    d = ippmin[pre_v.series_uid] - ippmin[post_v.series_uid]  # IPPpre - IPPpost
    sx, sy, sz = pre_v.spacing
    dk = float(np.dot(d, n)) / sz   # out[k,j,i] = post[k+dk, j+dj, i+di]
    dj = float(np.dot(d, c)) / sy
    di = float(np.dot(d, r)) / sx
    if max(abs(dk), abs(dj), abs(di)) < 1e-3:
        return replace(post_v, origin=pre_v.origin), dk
    shifted = ndimage.shift(post_v.data.astype(np.float32), (-dk, -dj, -di),
                            order=1, mode="constant", cval=0.0)
    return replace(post_v, data=shifted, origin=pre_v.origin), dk


report = []
combined = []
ONLY = {"3": "RAT_3"}   # 원본 폴더명 → 출력 폴더명
for rat in sorted(os.listdir(DATA_DIR)):
    rat_dir = os.path.join(DATA_DIR, rat)
    if not os.path.isdir(rat_dir) or rat not in ONLY:
        continue
    rat = ONLY[rat]
    animal = rat                        # RAT1 / RAT1_2 / RAT1_3_GD / RAT2 / RAT3
    out_rat = os.path.join(OUT_DIR, rat)
    os.makedirs(out_rat, exist_ok=True)
    print(f"\n===== {rat} =====", flush=True)

    vols = _load_volumes(rat_dir, True)
    ippmin = series_min_ipp(rat_dir)

    slabs = []
    for ori in sorted({v.orientation for v in vols}):
        vs = sorted([v for v in vols if v.orientation == ori],
                    key=lambda v: v.sort_key())
        if vs[0].tp != 0:
            print(f"  [경고] {ori}: pre(tp=0) 없음 → 건너뜀", flush=True)
            continue
        pre_v, posts = vs[0], vs[1:]
        merged, dks = [pre_v], []
        for pv in posts:
            mv, dk = merge_to_pre_grid(pre_v, pv, ippmin)
            merged.append(mv); dks.append(dk)
        nz = pre_v.n_slices
        lo = max(0, int(np.ceil(max(0.0, *[-k for k in dks]))))
        hi = min(nz - 1, int(np.floor(min(float(nz - 1), *[nz - 1 - k for k in dks]))))
        for pv, dk in zip(posts, dks):
            if abs(dk) > 1e-3:
                print(f"  {ori} post{pv.tp}: pre 격자로 리샘플 (면외 {dk:+.2f} slice)", flush=True)
        print(f"  {ori}: 전 시점 유효 슬라이스 = {lo}~{hi} (0-index, 총 {nz})", flush=True)
        report.append({"rat": rat, "orientation": ori, "n_slices": nz,
                       "valid_lo": lo, "valid_hi": hi,
                       "max_shift_slice": round(max([abs(k) for k in dks] or [0.0]), 2)})
        slabs.append(Slab(volumes=merged, orientation=ori, animal=animal))

    slabs = [register_slab(s, "inplane", True, verbose=True) for s in slabs]

    curves = [enhancement_curve(s) for s in slabs]
    csv_path = os.path.join(out_rat, "enhancement_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["label", "orientation", "n_frames", "roi_mm3", "peak_pct",
                    "time_to_peak_frame", "washin", "auc"])
        for s, cv in zip(slabs, curves):
            row = [cv["label"], s.orientation, cv["n_frames"], cv["roi_mm3"],
                   cv["peak_pct"], cv["time_to_peak_frame"],
                   cv["washin_pct_per_unit"], cv["auc_pct"]]
            w.writerow(row)
            combined.append([rat] + row)
    plot_curves(curves, os.path.join(out_rat, "all_curves.png"))

    for s in slabs:
        tag = s.label()
        montage_anatomy(s, "pre", os.path.join(out_rat, f"{tag}_preGd_anatomy.png"))
        montage_anatomy(s, "peak", os.path.join(out_rat, f"{tag}_peakGd_anatomy.png"))
        montage_heatmap(s, os.path.join(out_rat, f"{tag}_heatmap.png"))

        d_ind = os.path.join(out_rat, s.orientation, "individual")
        d_hm = os.path.join(out_rat, s.orientation, "heatmap_rows")
        d_sub = os.path.join(out_rat, s.orientation, "subtraction_rows")
        for d in (d_ind, d_hm, d_sub):
            os.makedirs(d, exist_ok=True)

        wl = np.percentile(s.volumes[-1].data, [1, 99])
        for z in range(s.n_slices):
            save_frame_slices(s, z, d_ind, wl=wl)
            heatmap_frames_row(s, z, os.path.join(d_hm, f"{tag}_slice{z:02d}_heatmap_row.png"))
            subtraction_frames_row(s, z, os.path.join(d_sub, f"{tag}_slice{z:02d}_subtraction_row.png"))
        print(f"  {tag}: 전체 {s.n_slices}슬라이스 저장 완료", flush=True)

with open(os.path.join(OUT_DIR, f"enhancement_summary_all_{TAG_DAY}_RAT_3.csv"), "w",
          newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["rat", "label", "orientation", "n_frames", "roi_mm3", "peak_pct",
                "time_to_peak_frame", "washin", "auc"])
    w.writerows(combined)

with open(os.path.join(OUT_DIR, f"valid_slice_ranges_{TAG_DAY}_RAT_3.csv"), "w",
          newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["rat", "orientation", "n_slices", "valid_lo", "valid_hi", "max_shift_slice"])
    for r in report:
        w.writerow([r["rat"], r["orientation"], r["n_slices"],
                    r["valid_lo"], r["valid_hi"], r["max_shift_slice"]])

print("\n완료", flush=True)
