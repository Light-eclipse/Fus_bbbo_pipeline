# -*- coding: utf-8 -*-
"""면외 이동을 보정하면 음의 증강이 사라지는지 본다.

`negchange_diag.py`가 추정한 면외 이동량으로 POST를 슬라이스 방향으로 옮긴 뒤,
뇌실질에서 10% 넘게 떨어진 화소 비율을 다시 잰다. 이동 보정 후 대조 개체 수준으로
떨어지면 그 감소는 움직임 인공물이다. 국소 덩어리가 남으면 출혈·부종 후보로 본다.

1 mm 슬라이스를 선형 보간하므로 흐려진다. 보정 후 남는 것도 보간 오차일 수 있어,
남은 후보는 그림으로 따로 확인한다.

실행: python negchange_shiftfix.py
"""
import os
import csv
import numpy as np
from scipy import ndimage

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIAG = os.path.join(R, "MRI 분석 코드", "negchange_diag_result.csv")
DARK_FRAC = 0.85
THR = -0.10
MIN_MM3 = 0.10


def brain_mask(pre, tis):
    med = float(np.median(pre[tis]))
    bright = ndimage.binary_opening(tis & (pre >= DARK_FRAC * med), np.ones((1, 3, 3)))
    lab, n = ndimage.label(bright)
    sizes = ndimage.sum(bright, lab, range(1, n + 1))
    br = lab == (int(np.argmax(sizes)) + 1)
    return np.stack([ndimage.binary_fill_holes(br[z]) for z in range(br.shape[0])])


rows = list(csv.DictReader(open(DIAG, encoding="utf-8-sig")))
out_rows = []
print(f"{'날짜':9}{'개체':4}{'계열':4}{'방향':9}{'이동':>6}{'  감소% 전→후':>16}"
      f"{'  국소덩어리(≥0.1mm³) 전→후':>26}{'  최대 덩어리 후':>16}")
for x in rows:
    date, rat, kind, ori = x["date"], x["rat"], x["kind"], x["orientation"]
    dz = float(x["throughplane_shift_slice"])
    d = np.load(os.path.join(R, date, "2. 분석 결과", f"RAT {rat}", "corrected_volumes.npz"))
    pre = d[f"{kind}_{ori}_PRE"].astype(float)
    posts = sorted([k for k in d.files if k.startswith(f"{kind}_{ori}_POST")],
                   key=lambda k: int(k.rsplit("POST", 1)[1] or 1))
    post = d[posts[-1]].astype(float)
    sp = d[f"{kind}_{ori}_spacing"]
    vox = abs(float(np.prod(sp)))
    br0 = brain_mask(pre, d[f"{kind}_{ori}_analysis_mask"])
    br = ndimage.binary_erosion(br0, np.ones((1, 9, 9)))
    # 이동 보정 후 경계 슬라이스는 외삽이 되므로 뺀다
    nz = pre.shape[0]
    valid = np.ones(nz, bool)
    k = int(np.ceil(abs(dz)))
    if dz > 0:
        valid[:k] = False
    elif dz < 0:
        valid[nz - k:] = False
    brv = br & valid[:, None, None]
    ps = ndimage.gaussian_filter(pre, (0, 1, 1))
    res = []
    for q in (post, ndimage.shift(post, (dz, 0, 0), order=1, mode="nearest")):
        qs = ndimage.gaussian_filter(q, (0, 1, 1))
        c = qs / np.maximum(ps, 1e-6) - 1
        dec = (c < THR) & brv
        lab, n = ndimage.label(dec)
        sizes = ndimage.sum(dec, lab, range(1, n + 1)) * vox if n else np.array([0.0])
        big = sizes[sizes >= MIN_MM3]
        res.append((float(dec.sum() / max(brv.sum(), 1) * 100), int(big.size),
                    float(big.sum()), float(sizes.max())))
    out_rows.append([date, rat, kind, ori, dz, round(res[0][0], 2), round(res[1][0], 2),
                     res[0][1], res[1][1], round(res[0][2], 2), round(res[1][2], 2),
                     round(res[1][3], 3)])
    print(f"{date:9}{rat:4}{kind:4}{ori:9}{dz:+6.1f}   {res[0][0]:6.2f}→{res[1][0]:6.2f}%"
          f"   {res[0][1]:4d}개 {res[0][2]:6.2f}→{res[1][1]:4d}개 {res[1][2]:6.2f} mm³"
          f"   {res[1][3]:7.3f} mm³")
with open(os.path.join(R, "MRI 분석 코드", "negchange_shiftfix_result.csv"), "w", newline="",
          encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["date", "rat", "kind", "orientation", "shift_slice", "dec_pct_before",
                "dec_pct_after", "n_clusters_before", "n_clusters_after", "cluster_mm3_before",
                "cluster_mm3_after", "largest_cluster_after_mm3"])
    w.writerows(out_rows)
