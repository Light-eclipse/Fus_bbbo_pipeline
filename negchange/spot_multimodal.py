# -*- coding: utf-8 -*-
"""9/22 저신호 점이 출혈과 맞는지 시퀀스별 대비로 확인한다.

출혈(탈산소 헤모글로빈·헤모시데린)은 자화율 효과라 TE가 길고 재초점이 없을수록 크게
어두워진다(blooming). 이 데이터에서 민감도 순서는
    IVIM SE-EPI b0 (TE 26 ms, EPI 판독)  >  GRE (TE 4 ms)  >  FSE (TE 7.4, 스핀에코)
정맥 단면도 어둡지만, 같은 논리로 EPI에서 가장 어둡다. 그래서 출혈과 정맥을 가르는 것은
대비 순서가 아니라 **모양**(축상·관상 모두 둥근지)과 **대측 같은 자리**다.

측정: 점 중심 반경 0.5 mm 원판 평균 / 반경 1.0~2.0 mm 고리 평균 − 1 (음수 = 어두움).
같은 값을 좌우 거울 위치(뇌 마스크 x 중심 기준 반사)에서도 잰다.

실행: python spot_multimodal.py
"""
import os
import glob
import csv
import numpy as np
import pydicom
import nibabel as nib
from scipy import ndimage

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(R, "26-09-22", "2. 분석 결과")
DATA = os.path.join(R, "26-09-22", "260922_BBBO_200G")


def geom(folder):
    fs = glob.glob(os.path.join(folder, "*.dcm"))
    ds = [pydicom.dcmread(f, stop_before_pixels=True) for f in fs]
    iop = np.array([float(x) for x in ds[0].ImageOrientationPatient])
    r, c = iop[:3], iop[3:]
    n = np.cross(r, c)
    ipp0 = min((np.array([float(v) for v in x.ImagePositionPatient]) for x in ds),
               key=lambda p: float(p @ n))
    return dict(ipp0=ipp0, r=r, c=c, n=n, ps=float(ds[0].PixelSpacing[0]),
                sz=float(getattr(ds[0], "SpacingBetweenSlices", 1.0) or 1.0))


def p2v(g, P):
    d = P - g["ipp0"]
    return d @ g["n"] / g["sz"], d @ g["c"] / g["ps"], d @ g["r"] / g["ps"]


def contrast(img2d, ci, cj, ps):
    yy, xx = np.mgrid[:img2d.shape[0], :img2d.shape[1]]
    r = np.hypot(yy - ci, xx - cj) * ps
    core = img2d[r <= 0.5]
    ring = img2d[(r > 1.0) & (r <= 2.0)]
    if core.size == 0 or ring.size == 0:
        return np.nan
    return float(core.mean() / ring.mean() - 1) * 100


def read_nii(f):
    with open(f, "rb") as fh:
        h = nib.Nifti1Header.from_fileobj(fh)
        return np.asarray(h.data_from_fileobj(fh), np.float64)


# 검사할 점: 9/22 개체 1 감소 #1, 개체 4 후보 #6 (10%p), 대조로 개체 4 후보 #2(정중선 혈관)
spots = []
for x in csv.DictReader(open(os.path.join(R, "MRI 분석 코드", "negchange_localize_result.csv"),
                             encoding="utf-8-sig")):
    if x["date"] == "26-09-22" and x["rat"] == "1" and float(x["max_vol_mm3"]) > 20:
        spots.append(("1", "개체1 감소#1 (PRE 저신호점)", np.array([float(v) for v in x["patient_xyz_mm"].split()])))
for x in csv.DictReader(open(os.path.join(BASE, "lesion_summary_pp10.csv"), encoding="utf-8-sig")):
    if x["rat_folder"] == "4" and x["confidence"].startswith("높음") and x["lesion"] in ("2", "6"):
        spots.append(("4", f"개체4 후보#{x['lesion']}", np.array([float(v) for v in x["patient_xyz_mm"].split()])))

print(f"{'점':22}{'시퀀스':14}{'z':>5}{'점 대비%':>10}{'  거울 대비%':>12}")
for rat, name, P in spots:
    d = np.load(os.path.join(BASE, f"RAT {rat}", "corrected_volumes.npz"))
    iv = np.load(os.path.join(BASE, f"RAT {rat}", "ivim", "ivim_maps.npz"))
    t2 = np.load(os.path.join(BASE, f"RAT {rat}", "t2", "t2_maps.npz"))
    shift = {x["rat"]: (int(x["shift_rows"]), int(x["shift_cols"]))
             for x in csv.DictReader(open(os.path.join(BASE, "ivim_summary.csv"), encoding="utf-8-sig"))}[rat]
    gA = geom(glob.glob(os.path.join(DATA, rat, "*GRE_AX_PRE*"))[0])
    # 좌우 중심: GRE 축상 뇌 마스크의 환자 x 평균 대신, 조직 마스크 열 중심을 환자 x로 환산
    m = d["gre_axial_analysis_mask"]
    zc, ic, jc = np.argwhere(m).mean(0)
    xmid = float((gA["ipp0"] + jc * gA["ps"] * gA["r"] + ic * gA["ps"] * gA["c"] + zc * gA["sz"] * gA["n"])[0])
    Pm = P.copy()
    Pm[0] = 2 * xmid - P[0]
    rows_out = []
    # GRE / FSE 축상·관상, PRE와 마지막 POST
    for kind, post in (("gre", "POST5"), ("fse", "POST1")):
        for ori, tag in (("axial", "AX"), ("coronal", "COR")):
            g = geom(glob.glob(os.path.join(DATA, rat, f"*{kind.upper()}_{tag}_PRE*"))[0])
            for tp in ("PRE", post):
                vol = d[f"{kind}_{ori}_{tp}"].astype(float)
                vals = []
                for Q in (P, Pm):
                    z, i, j = p2v(g, Q)
                    zi = int(round(z))
                    vals.append(contrast(vol[zi], i, j, g["ps"]) if 0 <= zi < vol.shape[0] else np.nan)
                print(f"{name:22}{kind.upper()+' '+tag+' '+tp:14}{int(round(p2v(g, P)[0])):5d}"
                      f"{vals[0]:10.1f}{vals[1]:12.1f}")
    # IVIM b0 (EPI, Gd 전)
    gI = dict(ipp0=iv["ipp0"], r=iv["row_dir"], c=iv["col_dir"], n=iv["slice_dir"],
              ps=float(iv["pixel_mm"]), sz=float(iv["slice_mm"]))
    S0 = iv["S0"]
    vals = []
    for Q in (P, Pm):
        z, i, j = p2v(gI, Q)
        zi = int(round(z))
        vals.append(contrast(S0[zi], i + shift[0], j + shift[1], gI["ps"]) if 0 <= zi < 14 else np.nan)
    print(f"{name:22}{'IVIM EPI b0':14}{int(round(p2v(gI, P)[0])):5d}{vals[0]:10.1f}{vals[1]:12.1f}")
    # MREPT 에코1, 에코8 (GRE 축상 128 격자 + 보정 이동)
    E = t2["echoes"]
    sh = t2["shift"]
    for e in (0, 7):
        vals = []
        for Q in (P, Pm):
            z, i, j = p2v(gA, Q)
            zi = int(round(z))
            vals.append(contrast(E[e][zi], i / 2 + sh[0], j / 2 + sh[1], gA["ps"] * 2)
                        if 0 <= zi < 14 else np.nan)
        print(f"{name:22}{'MREPT 에코'+str(e+1):14}{int(round(p2v(gA, P)[0])):5d}{vals[0]:10.1f}{vals[1]:12.1f}")
    print("-" * 64)
