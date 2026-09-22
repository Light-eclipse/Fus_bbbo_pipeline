# -*- coding: utf-8 -*-
"""뇌실질에서 **실제로 신호가 떨어진 곳**(POST < PRE, 음의 증강)을 찾는다.

왜 이 방식인가
--------------
국소 배경 대비 결손(`hypo_scan.py`)은 Gd 후 표면 경막·혈관이 밝아지면 그 안쪽 조직이
상대적으로 어둡게 잡혀, 뇌 표면 아래 약 1 mm에 "새로 생긴" 가짜 결손이 모든 날짜에서
생겼다. T1 강조 영상에서 Gd 뒤 신호가 **절대적으로** 떨어지는 것은 정상 조직에서 일어나지
않으므로, PRE 대비 POST의 비율 자체를 본다(스케일 보정된 값끼리).

음의 증강이 생길 수 있는 원인은 여럿이다. 후보마다 판별 지표를 같이 낸다.
* 정합 잔차 → 조직 경계(PRE 기울기가 큰 곳)에 생기고 반대편에 양의 띠가 짝으로 붙는다.
  **경계 점수** = 후보 안 |∇PRE| 평균 / 뇌 전체 |∇PRE| 중앙값. 1보다 훨씬 크면 경계.
* 출혈(자화율) → GRE(TE 4 ms)에서는 약하고 EPI·긴 TE에서 크게 번진다.
* 부종 → T1 연장으로 T1 강조 신호가 준다.
* 전역 드리프트·스케일 잔차 → 국소가 아니라 전체가 같이 움직인다.

실행: python negchange_scan.py "<날짜>\\2. 분석 결과"
"""
import os
import sys
import csv
import glob
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mri_dce import _set_korean_font
_set_korean_font()
plt.rcParams["axes.unicode_minus"] = False

DARK_FRAC = 0.85
ERODE_PX = 4
THR = -0.10          # 이보다 떨어지면 후보
MIN_MM3 = 0.10


def brain_mask(pre, tis):
    med = float(np.median(pre[tis]))
    bright = ndimage.binary_opening(tis & (pre >= DARK_FRAC * med), np.ones((1, 3, 3)))
    lab, n = ndimage.label(bright)
    sizes = ndimage.sum(bright, lab, range(1, n + 1))
    br = lab == (int(np.argmax(sizes)) + 1)
    return np.stack([ndimage.binary_fill_holes(br[z]) for z in range(br.shape[0])])


def main(base):
    rats = sorted(os.path.basename(p)[4:] for p in glob.glob(os.path.join(base, "RAT *")))
    rows, glob_rows = [], []
    for rat in rats:
        f = os.path.join(base, f"RAT {rat}", "corrected_volumes.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        out = os.path.join(base, f"RAT {rat}", "negchange")
        os.makedirs(out, exist_ok=True)
        for kind in ("gre", "fse"):
            for ori in ("axial", "coronal"):
                if f"{kind}_{ori}_PRE" not in d.files:
                    continue
                posts = sorted([x for x in d.files if x.startswith(f"{kind}_{ori}_POST")],
                               key=lambda x: int(x.rsplit("POST", 1)[1] or 1))
                last = posts[-1].split("_")[-1]
                pre = d[f"{kind}_{ori}_PRE"].astype(float)
                post = d[posts[-1]].astype(float)
                tis = d[f"{kind}_{ori}_analysis_mask"]
                sp = d[f"{kind}_{ori}_spacing"]
                br0 = brain_mask(pre, tis)
                br = ndimage.binary_erosion(br0, np.ones((1, 2 * ERODE_PX + 1, 2 * ERODE_PX + 1)))
                ps = ndimage.gaussian_filter(pre, (0, 1.0, 1.0))
                qs = ndimage.gaussian_filter(post, (0, 1.0, 1.0))
                c = np.where(br, qs / np.maximum(ps, 1e-6) - 1.0, np.nan)
                g = np.hypot(*np.gradient(ps, axis=(1, 2)))
                gmed = float(np.median(g[br]))
                depth = ndimage.distance_transform_edt(br0, sampling=(float(sp[2]), float(sp[1]),
                                                                      float(sp[0])))
                midcol = float(np.argwhere(br0)[:, 2].mean())
                vox = abs(float(np.prod(sp)))
                v = c[br]
                glob_rows.append([rat, kind, ori, last, round(float(np.median(v)) * 100, 2),
                                  round(float(np.percentile(v, 5)) * 100, 2),
                                  round(float(np.percentile(v, 1)) * 100, 2),
                                  round(float((v < THR).mean()) * 100, 3)])
                cand = np.nan_to_num(c, nan=0.0) < THR
                lab, n = ndimage.label(cand)
                if n:
                    sizes = ndimage.sum(cand, lab, range(1, n + 1))
                    for k in np.where(sizes * vox >= MIN_MM3)[0]:
                        sel = lab == (k + 1)
                        pts = np.argwhere(sel)
                        zc, rc, cc = pts.mean(0)
                        xy = pts[:, 1:].astype(float) * np.array([sp[1], sp[0]])
                        ev = np.sort(np.linalg.eigvalsh(np.cov(xy.T))) if len(xy) > 2 else np.ones(2)
                        # 반대 부호 짝: 후보를 2 mm 넓힌 띠 안의 양의 증강 최대
                        ring = ndimage.binary_dilation(sel, np.ones((1, 2 * int(2 / sp[0]) + 1,
                                                                        2 * int(2 / sp[0]) + 1))) & ~sel & br
                        rows.append(dict(
                            rat=rat, kind=kind, ori=ori, post=last,
                            vol_mm3=round(float(sizes[k] * vox), 3),
                            n_slice=len(set(pts[:, 0].tolist())),
                            z=round(float(zc), 1), row=int(round(rc)), col=int(round(cc)),
                            change_mean_pct=round(float(c[sel].mean()) * 100, 1),
                            change_min_pct=round(float(c[sel].min()) * 100, 1),
                            ring_max_pct=round(float(np.nanmax(c[ring])) * 100, 1) if ring.any() else "",
                            edge_score=round(float(g[sel].mean() / gmed), 2),
                            elong=round(float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9))), 2),
                            depth_mm=round(float(depth[sel].mean()), 2),
                            mid_mm=round(abs(cc - midcol) * float(sp[0]), 2),
                            side="L" if cc < midcol else "R"))
                # 몽타주
                zs = [z for z in range(pre.shape[0]) if br[z].sum() > 200]
                fig, axes = plt.subplots(3, len(zs), figsize=(1.7 * len(zs), 5.6), squeeze=False)
                wl = np.percentile(pre[br0], [1, 99.5])
                for j, z in enumerate(zs):
                    axes[0, j].imshow(pre[z], cmap="gray", vmin=wl[0], vmax=wl[1])
                    axes[1, j].imshow(post[z], cmap="gray", vmin=wl[0], vmax=wl[1])
                    axes[2, j].imshow(c[z] * 100, cmap="RdBu_r", vmin=-30, vmax=30)
                    axes[0, j].set_title(f"z={z}", fontsize=7)
                for a in axes.ravel():
                    a.set_xticks([])
                    a.set_yticks([])
                axes[0, 0].set_ylabel("PRE", fontsize=8)
                axes[1, 0].set_ylabel(last, fontsize=8)
                axes[2, 0].set_ylabel("변화 % (파랑=감소)", fontsize=8)
                fig.suptitle(f"RAT {rat} {kind} {ori} — 뇌실질 PRE→{last} 신호 변화 (침식한 뇌 마스크 안)",
                             fontsize=10)
                fig.tight_layout()
                fig.savefig(os.path.join(out, f"negchange_{kind}_{ori}.png"), dpi=110, bbox_inches="tight")
                plt.close(fig)
    with open(os.path.join(base, "negchange_candidates.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        if rows:
            w.writerow(list(rows[0].keys()))
            for r in sorted(rows, key=lambda r: (r["rat"], r["kind"], r["ori"], -r["vol_mm3"])):
                w.writerow(list(r.values()))
    with open(os.path.join(base, "negchange_global.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["rat", "kind", "orientation", "post", "median_change_pct", "p5_pct", "p1_pct",
                    "frac_below_thr_pct"])
        w.writerows(glob_rows)
    return rows, glob_rows


if __name__ == "__main__":
    rows, g = main(sys.argv[1])
    print(f"후보 {len(rows)}개")
    for r in g:
        print("  ", r)
