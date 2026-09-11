# -*- coding: utf-8 -*-
"""세 개체의 FSE 증강을 같은 스케일로 나란히 본다 (국소 누출 탐색용).

전조직 증강 곡선은 두피·근육이 지배하므로 뇌실질 누출을 볼 수 없다. 여기서는
뇌 중심부만 잘라내고 상한을 20%로 낮춰, 국소 누출이 있으면 주변보다 밝은 덩어리로
드러나게 한다. 6화소 가우시안 평활은 화소 잡음을 눌러 덩어리를 보이게 하려는 것이고,
마스크 밖 화소가 섞이지 않도록 마스크 가중으로 정규화한다.

실행: python "_compare_fse.py"
"""
import os
import sys
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(BASE)), "MRI 분석 코드"))
from mri_dce import _set_korean_font
_set_korean_font()

RATS = ("RAT 2", "RAT 3", "RAT 4")
VMAX = 20

for kind, ori, nz in (("fse", "coronal", 10), ("fse", "axial", 14),
                      ("gre", "coronal", 10), ("gre", "axial", 14)):
    fig, axes = plt.subplots(len(RATS), nz, figsize=(1.55 * nz, 1.8 * len(RATS)))
    im = None
    for r, rat in enumerate(RATS):
        d = np.load(os.path.join(BASE, rat, "corrected_volumes.npz"))
        pre = d[f"{kind}_{ori}_PRE"].astype(float)
        posts = sorted([x for x in d.files if x.startswith(f"{kind}_{ori}_POST")],
                       key=lambda x: int(x.rsplit("POST", 1)[1] or 1))
        post = d[posts[-1]].astype(float)
        m = d[f"{kind}_{ori}_analysis_mask"]
        enh = np.where(m, (post - pre) / (pre + 1e-6) * 100.0, 0.0)
        # 마스크 가중 평활: 마스크 밖 0이 섞여 값이 낮아지지 않게 한다
        num = ndimage.gaussian_filter(enh, (0, 1.2, 1.2))
        den = ndimage.gaussian_filter(m.astype(float), (0, 1.2, 1.2))
        enh = np.where(den > 0.5, num / np.maximum(den, 1e-6), np.nan)
        idx = np.argwhere(m.any(0))
        r0, c0 = idx.min(0); r1, c1 = idx.max(0) + 1
        dr, dc = int((r1 - r0) * 0.24), int((c1 - c0) * 0.24)
        r0, r1, c0, c1 = r0 + dr, r1 - dr, c0 + dc, c1 - dc
        for z in range(nz):
            im = axes[r, z].imshow(enh[z, r0:r1, c0:c1], cmap="jet", vmin=0, vmax=VMAX)
            axes[r, z].set_xticks([]); axes[r, z].set_yticks([])
            if r == 0:
                axes[r, z].set_title(f"z={z}", fontsize=8)
            if z == 0:
                axes[r, z].set_ylabel(rat, fontsize=9)
    fig.suptitle(f"{kind.upper()} {ori} 증강% (상한 {VMAX}%, 평활, 뇌 중심 크롭) — "
                 f"국소 누출이 있으면 주변보다 밝은 덩어리로 보인다", fontsize=11)
    fig.subplots_adjust(right=0.93)
    fig.colorbar(im, cax=fig.add_axes([0.945, 0.15, 0.008, 0.7]), label="%")
    out = os.path.join(BASE, f"compare_{kind}_{ori}_vmax{VMAX}.png")
    fig.savefig(out, dpi=115, bbox_inches="tight")
    plt.close(fig)
    print("저장:", os.path.basename(out), flush=True)
