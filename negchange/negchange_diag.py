# -*- coding: utf-8 -*-
"""음의 증강이 국소 병변(출혈 등)인지, 면외 이동·감도 분포 변화 같은 전역 요인인지 가른다.

1. **면외 이동 추정**: PRE와 POST의 기울기 크기 영상(대비 차이에 덜 민감)을 뇌 마스크 안에서
   슬라이스 방향으로 −1.5~+1.5 slice 평행이동하며 상관을 본다. 정합(`register_slab`,
   inplane)은 면내만 보정하므로 면외 이동은 남아 있다.
2. **매끄러운 장 분리**: 뇌 안 log(POST/PRE)에 3차원 2차 다항식(10항)을 맞춘다.
   코일 감도·B1 분포 변화나 넓은 영역의 변화는 이 매끄러운 장이 대부분 설명한다.
   장을 뺀 잔차에서 −10% 아래로 떨어지는 화소가 **국소** 신호 감소다. 출혈이라면 여기 남아야 한다.

실행: python negchange_diag.py
"""
import os
import sys
import csv
import numpy as np
from scipy import ndimage

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASES = [("26-09-11", r) for r in ("2", "3", "4")] + \
        [("26-09-15", r) for r in ("1", "3", "4")] + [("26-09-22", r) for r in ("1", "4")]
DARK_FRAC = 0.85


def brain_mask(pre, tis):
    med = float(np.median(pre[tis]))
    bright = ndimage.binary_opening(tis & (pre >= DARK_FRAC * med), np.ones((1, 3, 3)))
    lab, n = ndimage.label(bright)
    sizes = ndimage.sum(bright, lab, range(1, n + 1))
    br = lab == (int(np.argmax(sizes)) + 1)
    return np.stack([ndimage.binary_fill_holes(br[z]) for z in range(br.shape[0])])


def gradmag(v):
    s = ndimage.gaussian_filter(v, (0, 1.5, 1.5))
    return np.hypot(*np.gradient(s, axis=(1, 2)))


rows = []
print(f"{'날짜':9}{'개체':4}{'계열':4}{'방향':9}{'면외이동':>9}{'상관(0→최적)':>16}"
      f"{'  감소<-10%':>11}{'장R²':>7}{'장 뺀 뒤':>9}")
for date, rat in CASES:
    f = os.path.join(R, date, "2. 분석 결과", f"RAT {rat}", "corrected_volumes.npz")
    d = np.load(f)
    for kind in ("gre", "fse"):
        for ori in ("axial", "coronal"):
            pre = d[f"{kind}_{ori}_PRE"].astype(float)
            posts = sorted([x for x in d.files if x.startswith(f"{kind}_{ori}_POST")],
                           key=lambda x: int(x.rsplit("POST", 1)[1] or 1))
            post = d[posts[-1]].astype(float)
            tis = d[f"{kind}_{ori}_analysis_mask"]
            br0 = brain_mask(pre, tis)
            br = ndimage.binary_erosion(br0, np.ones((1, 9, 9)))
            # 1. 면외 이동
            gp, gq = gradmag(pre), gradmag(post)
            m = ndimage.binary_dilation(br0, np.ones((1, 5, 5)))
            best, r0 = (0.0, -1.0), None
            for dz in np.arange(-1.5, 1.51, 0.1):
                sh = ndimage.shift(gq, (dz, 0, 0), order=1, mode="nearest")
                k = m.copy()
                if dz > 0:
                    k[:int(np.ceil(dz))] = False
                elif dz < 0:
                    k[pre.shape[0] - int(np.ceil(-dz)):] = False
                r = float(np.corrcoef(gp[k], sh[k])[0, 1])
                if abs(dz) < 1e-9:
                    r0 = r
                if r > best[1]:
                    best = (float(dz), r)
            # 2. 매끄러운 장
            ps = ndimage.gaussian_filter(pre, (0, 1, 1))
            qs = ndimage.gaussian_filter(post, (0, 1, 1))
            y = np.log(np.maximum(qs, 1e-6) / np.maximum(ps, 1e-6))[br]
            zz, yy, xx = np.nonzero(br)
            zz = (zz - zz.mean()) / (zz.std() + 1e-9)
            yy = (yy - yy.mean()) / (yy.std() + 1e-9)
            xx = (xx - xx.mean()) / (xx.std() + 1e-9)
            X = np.stack([np.ones_like(zz), zz, yy, xx, zz * zz, yy * yy, xx * xx,
                          zz * yy, zz * xx, yy * xx], 1).astype(float)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            fit = X @ coef
            ss_res = float(((y - fit) ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2 = 1 - ss_res / max(ss_tot, 1e-12)
            dec = float((np.exp(y) - 1 < -0.10).mean()) * 100
            dec_res = float((np.exp(y - fit) - 1 < -0.10).mean()) * 100
            rows.append([date, rat, kind, ori, round(best[0], 1), round(r0, 4), round(best[1], 4),
                         round(dec, 2), round(r2, 3), round(dec_res, 2)])
            print(f"{date:9}{rat:4}{kind:4}{ori:9}{best[0]:+9.1f}   {r0:.4f}→{best[1]:.4f}"
                  f"{dec:10.2f}%{r2:7.3f}{dec_res:8.2f}%")
out = os.path.join(R, "MRI 분석 코드", "negchange_diag_result.csv")
with open(out, "w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["date", "rat", "kind", "orientation", "throughplane_shift_slice",
                "grad_corr_at_0", "grad_corr_best", "vox_decrease_gt10pct",
                "smooth_field_R2", "vox_decrease_after_smooth_removed"])
    w.writerows(rows)
print("\n저장:", out)
