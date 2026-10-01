# -*- coding: utf-8 -*-
"""다중획득 재현 병변을 GRE·FSE 축상 영상 위에 표시해 눈으로 확인하는 그림.

`lesion_summary.csv`(15%p)와 `lesion_summary_pp10.csv`(10%p)에서 신뢰 "높음"만 골라,
병변 중심(환자 좌표)을 각 계열의 축상 격자로 옮겨 가장 가까운 슬라이스를 그린다.
윗줄 축상, 아랫줄 관상. 각 줄: GRE PRE | POST5 | 증강% | FSE PRE | POST | 증강%.
원은 반경 1 mm(빨강)과 3 mm(노랑 점선).

자동 검출이 통과시킨 후보도 그림에서 정합 테두리나 정맥동으로 드러나는 경우가
있었으므로(26-09-11, 26-09-15), 결론은 이 그림을 보고 낸다.

실행: python "_lesion_figs.py"
"""
import os
import sys
import csv
import glob
import numpy as np
import pydicom
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))
DATA_ROOT = os.path.join(os.path.dirname(BASE), "261001_BBBO")
sys.path.insert(0, os.path.join(ROOT, "MRI 분석 코드"))
from mri_dce import _set_korean_font
_set_korean_font()
plt.rcParams["axes.unicode_minus"] = False


def geom(rat, pattern):
    d = glob.glob(os.path.join(DATA_ROOT, rat, pattern))[0]
    fs = glob.glob(os.path.join(d, "*.dcm"))
    ds = [pydicom.dcmread(f, stop_before_pixels=True) for f in fs]
    iop = np.array([float(x) for x in ds[0].ImageOrientationPatient])
    r, c = iop[:3], iop[3:]
    n = np.cross(r, c)
    ipp0 = min((np.array([float(v) for v in x.ImagePositionPatient]) for x in ds),
               key=lambda p: float(p @ n))
    return dict(ipp0=ipp0, r=r, c=c, n=n, ps=float(ds[0].PixelSpacing[0]),
                sz=float(getattr(ds[0], "SpacingBetweenSlices", 1.0) or 1.0))


def to_vox(g, P):
    d = P - g["ipp0"]
    return d @ g["n"] / g["sz"], d @ g["c"] / g["ps"], d @ g["r"] / g["ps"]


rows = []
for fn, src in (("lesion_summary.csv", "15%p"), ("lesion_summary_pp10.csv", "10%p")):
    fp = os.path.join(BASE, fn)
    if not os.path.exists(fp):
        continue
    for x in csv.DictReader(open(fp, encoding="utf-8-sig")):
        if x["confidence"].startswith("높음"):
            rows.append((src, x))

for src, x in rows:
    rat = x["rat_folder"]
    d = np.load(os.path.join(BASE, f"RAT {rat}", "corrected_volumes.npz"))
    P = np.array([float(v) for v in x["patient_xyz_mm"].split()])
    fig, axes = plt.subplots(2, 6, figsize=(21, 7.6))
    for row_, ori, tag in ((0, "axial", "AX"), (1, "coronal", "COR")):
        ax = axes[row_]
        for col, kind, post in ((0, "gre", "POST5"), (3, "fse", "POST1")):
            g = geom(rat, f"*{kind.upper()}_{tag}_PRE*")
            kz, ki, kj = to_vox(g, P)
            z = int(round(kz))
            pre = d[f"{kind}_{ori}_PRE"].astype(float)
            pst = d[f"{kind}_{ori}_{post}"].astype(float)
            m = d[f"{kind}_{ori}_analysis_mask"]
            if not 0 <= z < pre.shape[0]:
                for a in ax[col:col + 3]:
                    a.text(0.5, 0.5, f"{ori} 슬랩 밖 (z={kz:.1f})", ha="center",
                           transform=a.transAxes)
                    a.axis("off")
                continue
            enh = np.where(m[z], (pst[z] - pre[z]) / (pre[z] + 1e-6) * 100, np.nan)
            wl = np.percentile(pre[pre > 0], [2, 99.5])
            rad = 1.0 / g["ps"]
            for a, img, nm, kw in ((ax[col], pre[z], "PRE", dict(cmap="gray", vmin=wl[0], vmax=wl[1])),
                                   (ax[col + 1], pst[z], post, dict(cmap="gray", vmin=wl[0], vmax=wl[1])),
                                   (ax[col + 2], enh, "증강 %", dict(cmap="jet", vmin=0, vmax=60))):
                a.imshow(img, **kw)
                a.add_patch(Circle((kj, ki), rad, fill=False, ec="red", lw=1.4))
                a.add_patch(Circle((kj, ki), rad * 3, fill=False, ec="yellow", lw=0.6, ls="--"))
                a.set_title(f"{kind.upper()} {ori} z={z}  {nm}", fontsize=9)
                a.axis("off")
    fig.suptitle(f"폴더 {rat}  [{src}] 병변 {x['lesion']}  부피 {x['max_volume_mm3']} mm³  "
                 f"깊이 {x['depth_below_brain_surface_mm']} mm  순 누출 {x['mean_net_leakage_pp']} %p  "
                 f"획득 {x['acquisitions']}  {x['confidence']}   (빨강 원 반경 1 mm, 노랑 3 mm)",
                 fontsize=10)
    fig.tight_layout()
    out = os.path.join(BASE, f"RAT {rat}", "lesion_check")
    os.makedirs(out, exist_ok=True)
    fn = os.path.join(out, f"lesion_{src.replace('%', 'pct')}_{x['lesion']}.png")
    fig.savefig(fn, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print("저장:", os.path.relpath(fn, BASE))
