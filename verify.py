# -*- coding: utf-8 -*-
"""261001 BBBO — 분석 전 데이터 확인.

시리즈별 프로토콜·스케일 태그·촬영 시각·기하를 전부 찍어 본다. 분석에 들어가기 전에
확인할 것은 세 가지다.

1. PRE와 POST가 같은 프로토콜·같은 격자인가 (증강을 계산할 수 있는 조건)
2. 스케일 태그 (00E1,1001)가 시리즈마다 다른가 (Agilent FdfToDcm 재정규화)
3. 파일명과 실제 기하가 어긋난 시리즈가 있는가
   — 폴더 2에 `T2_star_AX-NO1` 이 들어 있다. 이름은 개체 1인데 폴더는 2다.

실행: python "_verify_1001.py"
"""
import os
import re
import csv
import glob
import numpy as np
import pydicom

ROOT = r"C:\Users\user\Desktop\대학원\6. 원자력의학원"
DATA = os.path.join(ROOT, "26-10-01", "261001_BBBO")
OUT = os.path.join(ROOT, "26-10-01", "2. 분석 결과")
SCALE_TAG = (0x00E1, 0x1001)
RATS = ("1", "2", "3")


def orient(iop):
    r, c = np.array(iop[:3], float), np.array(iop[3:], float)
    n = np.abs(np.cross(r, c))
    return ("sagittal", "coronal", "axial")[int(np.argmax(n))]


rows = []
for rat in RATS:
    rdir = os.path.join(DATA, rat)
    print("=" * 100)
    print(f"폴더 {rat}")
    print(f"{'시리즈':<42} {'n':>3} {'seq':<7} {'TR':>7} {'TE':>7} {'FA':>5} "
          f"{'NEX':>4} {'ETL':>4} {'mtx':>5} {'px':>6} {'방향':<9} {'시각':<8} {'스케일':>9}")
    for folder in sorted(os.listdir(rdir)):
        d = os.path.join(rdir, folder)
        if not os.path.isdir(d) or folder.endswith(".nii"):
            continue
        fs = sorted(glob.glob(os.path.join(d, "*.dcm")))
        if not fs:
            continue
        ds = pydicom.dcmread(fs[0], stop_before_pixels=True)
        iop = ds.get("ImageOrientationPatient")
        ipp = ds.get("ImagePositionPatient")
        try:
            sc = float(ds[SCALE_TAG].value)
        except Exception:
            sc = float("nan")
        o = orient(iop) if iop else "?"
        r = dict(rat=rat, series=folder.replace("_20261001_01.dcm", ""),
                 n=len(fs), seq=str(ds.get("SequenceName", "?")),
                 tr=float(ds.get("RepetitionTime", 0)),
                 te=float(ds.get("EchoTime", 0)),
                 fa=float(ds.get("FlipAngle", 0)),
                 nex=float(ds.get("NumberOfAverages", 0)),
                 etl=str(ds.get("EchoTrainLength", "-")),
                 mtx=int(ds.Rows), px=float(ds.PixelSpacing[0]),
                 ori=o, t=str(ds.get("ContentTime", "?"))[:6], scale=sc,
                 ipp=" ".join(f"{float(x):.3f}" for x in ipp) if ipp else "",
                 iop=" ".join(f"{float(x):.3f}" for x in iop) if iop else "")
        rows.append(r)
        print(f"{r['series']:<42} {r['n']:>3} {r['seq']:<7} {r['tr']:>7.1f} {r['te']:>7.3f} "
              f"{r['fa']:>5.0f} {r['nex']:>4.0f} {r['etl']:>4} {r['mtx']:>5} {r['px']:>6.3f} "
              f"{r['ori']:<9} {r['t']:<8} {r['scale']:>9.4f}")

with open(os.path.join(OUT, "series_protocol_all.csv"), "w", newline="",
          encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ---------------------------------------------------------------- 기하 점검
print("\n" + "=" * 100)
print("DCE 계열 PRE/POST 기하 일치 (증강 계산의 전제)")
for rat in RATS:
    sub = [r for r in rows if r["rat"] == rat]
    for kind in ("GRE", "FSE"):
        for ax in ("AX", "COR"):
            grp = [r for r in sub if re.search(rf"{kind}_{ax}_(PRE|POST)", r["series"])]
            if not grp:
                continue
            pre = [r for r in grp if "_PRE" in r["series"]]
            if not pre:
                print(f"  폴더 {rat} {kind} {ax}: PRE 없음")
                continue
            p = pre[0]
            bad = [r for r in grp if r["iop"] != p["iop"] or r["ipp"] != p["ipp"]
                   or r["tr"] != p["tr"] or r["te"] != p["te"] or r["mtx"] != p["mtx"]]
            tag = "일치" if not bad else "어긋남: " + ", ".join(b["series"][-18:] for b in bad)
            sc = sorted({round(r["scale"], 4) for r in grp})
            print(f"  폴더 {rat} {kind} {ax:<4} n={len(grp)}  {tag}   스케일 {sc}")

# --------------------------------------------- 폴더 2 의 T2_star_AX-NO1 점검
print("\n" + "=" * 100)
print("폴더 2 의 `T2_star_AX-NO1` — 이름은 개체 1, 폴더는 2")
tgt = [r for r in rows if "T2_star" in r["series"]]
for r in tgt:
    print(f"  폴더 {r['rat']}  {r['series']:<34} IPP [{r['ipp']}]  시각 {r['t']}")
print("  같은 폴더의 GRE AX PRE 기준 IPP:")
for rat in RATS:
    g = [r for r in rows if r["rat"] == rat and re.search(r"GRE_AX_PRE", r["series"])]
    if g:
        print(f"    폴더 {rat}  [{g[0]['ipp']}]")

# ------------------------------------------------------------- IVIM b값 목록
print("\n" + "=" * 100)
print("IVIM b값 (폴더마다 다르다)")
for rat in RATS:
    bs = sorted(int(m.group(1)) for r in rows if r["rat"] == rat
                for m in [re.search(r"IVIM-EPI_B(\d+)", r["series"])] if m)
    print(f"  폴더 {rat}  {len(bs)}개  {bs}")
print("\n저장:", os.path.join(OUT, "series_protocol_all.csv"))
