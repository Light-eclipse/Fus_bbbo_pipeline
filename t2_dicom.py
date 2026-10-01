# -*- coding: utf-8 -*-
"""261001 — T2 / T2* 를 DICOM 에서 직접 적합한다.

이번 날짜에 달라진 점
--------------------
26-09-22 까지 다중에코는 NIfTI(`MREPT`) 로만 왔다. qform 쿼터니언이 비정규라
`nib.load` 가 죽고 sform 은 0 이라 **기하 정보가 없어서** GRE 와 상관으로 방향을
찾아야 했다. 이번에는 `T2_MAP`(mems, 스핀에코)과 `T2_star_AX`(mgems, 그래디언트
에코)가 **DICOM 으로** 들어왔다. IPP·IOP·화소 크기가 다 있으므로 기하 추정이
필요 없다. 실제로 폴더별 IPP 가 같은 폴더 GRE AX PRE 와 소수점까지 같다.

남은 가정 하나
-------------
에코별 EchoTime 이 여전히 전부 첫 에코 값이다(mems 10.0, mgems 3.0). 에코를
구분하는 태그는 `EchoNumbers` 뿐이다. 그래서 `TE_n = n x ΔTE` 로 둔다. ms 단위
T2/T2* 는 이 상수에 **선형 비례**하고, 에코당 R2 는 가정과 무관하다. 둘 다 낸다.

MREPT 와의 관계도 확인한다. MREPT nii 는 8에코, T2_MAP DICOM 은 10에코다. 같은
촬영의 일부인지 상관으로 본다.

짝·홀 에코
---------
26-09-22 에서 연속 에코 비가 0.889 / 0.708 / 0.852 로 번갈아 뛰었다(자극 에코로
짝수가 밝다). 여기서도 전/홀/짝 적합을 따로 내서 차이를 불확실성으로 적는다.

실행: python "_t2_dicom_1001.py"
"""
import os
import re
import sys
import csv
import glob
import numpy as np
import pydicom
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))
sys.path.insert(0, os.path.join(ROOT, "MRI 분석 코드"))
from mri_dce import _set_korean_font
_set_korean_font()
plt.rcParams["axes.unicode_minus"] = False

DATA = os.path.join(ROOT, "26-10-01", "261001_BBBO")
SCALE_TAG = (0x00E1, 0x1001)
RATS = sys.argv[1:] or ["1", "2", "3"]
DTE = {"mems": 10.0, "mgems": 3.0}      # 가정: TE_n = n x 이 값 (첫 에코 TE 와 같다고 봄)


def load_multiecho(d):
    """slice*image*echo*.dcm -> (echo, z, y, x), 스케일 태그 보정."""
    fs = sorted(glob.glob(os.path.join(d, "*.dcm")))
    idx = {}
    scale, seq, px, sz = None, None, None, None
    for f in fs:
        m = re.search(r"slice(\d+)image(\d+)echo(\d+)", os.path.basename(f))
        ds = pydicom.dcmread(f)
        z, e = int(m.group(1)) - 1, int(m.group(3)) - 1
        if scale is None:
            scale = float(ds[SCALE_TAG].value)
            seq = str(ds.get("SequenceName", "?"))
            px = float(ds.PixelSpacing[0])
            sz = float(ds.get("SliceThickness", 1.0))
        idx[(e, z)] = ds.pixel_array.astype(np.float64) / scale
    ne = max(k[0] for k in idx) + 1
    nz = max(k[1] for k in idx) + 1
    v = np.stack([np.stack([idx[(e, z)] for z in range(nz)]) for e in range(ne)])
    return v, dict(seq=seq, scale=scale, px=px, sz=sz, ne=ne, nz=nz)


def read_nii(f):
    with open(f, "rb") as fh:
        h = nib.Nifti1Header.from_fileobj(fh)
        a = np.asarray(h.data_from_fileobj(fh), np.float64)
    sl = float(h["scl_slope"]) if h["scl_slope"] else 1.0
    return a / sl


def fit_logline(v, te, m):
    """log S = log S0 - TE/T2 를 화소마다 최소제곱. (R2_per_echo, T2_ms) 반환."""
    y = np.log(np.clip(v, 1e-6, None))
    x = np.asarray(te, float)
    x = x - x.mean()
    sl = (x[:, None, None, None] * (y - y.mean(0))).sum(0) / (x ** 2).sum()
    r2 = -sl                                    # 에코 1개당 감쇠율
    return r2, np.where(m, 1.0 / np.clip(r2, 1e-9, None), np.nan)


rows = []
for rat in RATS:
    rdir = os.path.join(DATA, rat)
    print("=" * 84)
    print(f"폴더 {rat}")
    for tag, label in (("T2_MAP", "T2 (mems, 스핀에코)"),
                       ("T2_star", "T2* (mgems, 그래디언트에코)")):
        ds_dirs = [d for d in os.listdir(rdir) if tag in d and d.endswith(".dcm")]
        if not ds_dirs:
            print(f"  {label}: 없음")
            continue
        v, meta = load_multiecho(os.path.join(rdir, ds_dirs[0]))
        dte = DTE.get(meta["seq"], 1.0)
        te = np.arange(1, meta["ne"] + 1) * dte
        # 조직 마스크 (에코1 상위)
        m1 = v[0]
        tis = ndimage.binary_erosion(m1 > np.percentile(m1, 60),
                                     np.ones((1, 3, 3)), iterations=2)
        print(f"  {label}  {ds_dirs[0][:34]}  seq={meta['seq']}  "
              f"에코 {meta['ne']}  슬라이스 {meta['nz']}  화소 {meta['px']:.3f} mm")
        print(f"     에코별 조직 중앙 신호: "
              + " ".join(f"{np.median(v[e][tis]):.0f}" for e in range(meta["ne"])))
        ratios = [np.median(v[e + 1][tis]) / np.median(v[e][tis])
                  for e in range(meta["ne"] - 1)]
        print(f"     연속 에코 비: " + " ".join(f"{r:.3f}" for r in ratios))
        osc = np.std(ratios) / np.mean(ratios)
        print(f"     비의 변동 {osc*100:.1f}%  "
              + ("(짝·홀 진동 있음 -> 자극 에코)" if osc > 0.08 else "(단조)"))

        out = {}
        for nm, sel in (("전체", slice(None)), ("홀수", slice(0, None, 2)),
                        ("짝수", slice(1, None, 2))):
            r2, t2 = fit_logline(v[sel], te[sel], tis)
            med_r2 = float(np.median(r2[tis]))
            med_t2 = float(np.median(t2[tis]))
            out[nm] = (med_r2, med_t2)
            print(f"     {nm:<4} 적합: 에코당 R2 {med_r2:.4f}   "
                  f"T2{'*' if tag=='T2_star' else ''} {med_t2:.1f} ms  "
                  f"(ΔTE {dte} ms 가정)")
        spread = abs(out["홀수"][1] - out["짝수"][1]) / out["전체"][1] * 100
        print(f"     홀·짝 차이 {spread:.1f}%  <- 이 값을 불확실성으로 적는다")
        rows.append(dict(rat=rat, kind=tag, seq=meta["seq"], ne=meta["ne"],
                         dte_ms=dte, r2_all=out["전체"][0], ms_all=out["전체"][1],
                         ms_odd=out["홀수"][1], ms_even=out["짝수"][1],
                         odd_even_pct=spread, osc_pct=osc * 100))

    # ------------------------------------------- MREPT nii 가 같은 촬영인가
    nd = [d for d in os.listdir(rdir) if "MREPT" in d]
    if nd:
        mr = np.stack([read_nii(f) for f in
                       sorted(glob.glob(os.path.join(rdir, nd[0], "vol_*.nii")))])
        t2d = [d for d in os.listdir(rdir) if "T2_MAP" in d]
        if t2d:
            v, meta = load_multiecho(os.path.join(rdir, t2d[0]))
            # nii 는 (x,y,z), DICOM 은 (z,y,x)
            a = np.transpose(mr[0], (2, 1, 0))
            b = v[0]
            best = max(((float(np.corrcoef(a[:, ::sy, ::sx].ravel(),
                                           b.ravel())[0, 1]), sy, sx)
                        for sy in (1, -1) for sx in (1, -1)
                        if a[:, ::sy, ::sx].shape == b.shape), default=(0, 1, 1))
            print(f"  MREPT nii 에코1 vs T2_MAP DICOM 에코1 상관 {best[0]:+.4f} "
                  f"(y{'반전' if best[1]<0 else ''} x{'반전' if best[2]<0 else ''})")
            print("     -> 상관이 높으면 같은 촬영이고, DICOM 쪽을 쓰면 기하가 공짜다")

with open(os.path.join(BASE, "t2_summary_dicom.csv"), "w", newline="",
          encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\n저장:", os.path.join(BASE, "t2_summary_dicom.csv"))
