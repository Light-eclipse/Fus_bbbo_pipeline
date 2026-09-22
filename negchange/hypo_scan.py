# -*- coding: utf-8 -*-
"""뇌실질 국소 **저신호**(신호 감쇠) 검출 — 출혈 후보 탐색.

기존 분석(`local_focus`, `lateral_focus`)은 양의 증강(POST > PRE)만 찾았으므로 출혈처럼
**어두워지는** 병소는 검출 대상이 아니었다. 이 스크립트가 그 빈칸을 채운다.

판별해야 할 것
--------------
어두운 점은 출혈만이 아니다. GRE에서는 정맥(탈산소 헤모글로빈)·정맥동·뇌실 경계·공기
경계도 어둡다. 그래서 후보마다 다음을 같이 낸다.

* **PRE와 POST 각각의 결손**. 둘 다 어두우면 Gd 전부터 있던 것, POST에서만 어두우면 새로
  생긴 것. (이 연구실의 PRE는 Gd 전이지 FUS 전이 아닐 수 있다. FUS 시점은 헤더에 없다.)
* **길쭉함**(면내 2차 모멘트 비). 정맥은 관 모양이라 길다. 출혈은 둥글거나 불규칙한 덩어리.
* **정중선까지 거리**. 상시상정맥동 등.
* **깊이**(뇌 표면까지 거리). 피질 정맥은 표면에 붙어 있다.
* **좌우 반구 결손 부담**. 표적이 한쪽이면 출혈은 편측으로 몰린다. 정맥은 대체로 대칭이다.

방법
----
계열(GRE·FSE) × 방향(축상·관상) × 시점(PRE·마지막 POST)마다
  1. 뇌 마스크 = 조직 마스크에서 PRE < 0.85×조직중앙값 제거 → 최대 연결성분 → 슬라이스별
     구멍 메우기(뇌 안의 어두운 점은 구멍으로 메워져 포함된다) → 면내 4화소 침식(표면 제외).
  2. 영상을 1화소 가우시안으로 약하게 평활한 뒤, 마스크 가중 가우시안(시그마 1.2 mm)으로
     국소 배경을 만든다. 결손 = 1 − S / 배경.
  3. 결손 > DEF_THR 인 3차원 연결성분 중 MIN_VOX 이상을 후보로 낸다.
  4. 후보 자리의 결손을 반대 시점에서도 잰다.

실행: python hypo_scan.py "<날짜>\\2. 분석 결과"      (RAT * 폴더들을 모두 처리)
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
BG_SIGMA_MM = 1.2
DEF_THR = 0.20
MIN_VOX = 6
TOP_FIG = 6


def brain_mask(pre, tis):
    med = float(np.median(pre[tis]))
    bright = ndimage.binary_opening(tis & (pre >= DARK_FRAC * med), np.ones((1, 3, 3)))
    lab, n = ndimage.label(bright)
    if n == 0:
        return None
    sizes = ndimage.sum(bright, lab, range(1, n + 1))
    br = lab == (int(np.argmax(sizes)) + 1)
    return np.stack([ndimage.binary_fill_holes(br[z]) for z in range(br.shape[0])])


def deficit(img, br, sp):
    sig = BG_SIGMA_MM / float(sp[0])
    s = ndimage.gaussian_filter(img, (0, 1.0, 1.0))
    w = br.astype(float)
    num = ndimage.gaussian_filter(np.where(br, s, 0.0), (0, sig, sig))
    den = ndimage.gaussian_filter(w, (0, sig, sig))
    bg = num / np.maximum(den, 1e-6)
    return np.where(br, 1.0 - s / np.maximum(bg, 1e-6), 0.0)


def main(base):
    rats = sorted(os.path.basename(p)[4:] for p in glob.glob(os.path.join(base, "RAT *")))
    rows, burden = [], []
    for rat in rats:
        f = os.path.join(base, f"RAT {rat}", "corrected_volumes.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f)
        out = os.path.join(base, f"RAT {rat}", "hypo_scan")
        os.makedirs(out, exist_ok=True)
        for kind in ("gre", "fse"):
            for ori in ("axial", "coronal"):
                k = f"{kind}_{ori}_PRE"
                if k not in d.files:
                    continue
                posts = sorted([x for x in d.files if x.startswith(f"{kind}_{ori}_POST")],
                               key=lambda x: int(x.rsplit("POST", 1)[1] or 1))
                pre = d[k].astype(float)
                post = d[posts[-1]].astype(float)
                tis = d[f"{kind}_{ori}_analysis_mask"]
                sp = d[f"{kind}_{ori}_spacing"]
                br0 = brain_mask(pre, tis)
                if br0 is None:
                    continue
                br = ndimage.binary_erosion(br0, np.ones((1, 2 * ERODE_PX + 1, 2 * ERODE_PX + 1)))
                depth = ndimage.distance_transform_edt(br0, sampling=(float(sp[2]), float(sp[1]),
                                                                      float(sp[0])))
                midcol = float(np.argwhere(br0)[:, 2].mean())
                vox = abs(float(np.prod(sp)))
                D = {"PRE": deficit(pre, br, sp), posts[-1].split("_")[-1]: deficit(post, br, sp)}
                tag_post = posts[-1].split("_")[-1]
                # 좌우 반구 부담 (결손 > 문턱 화소 부피, 정중선 열 기준)
                for tp, Dm in D.items():
                    hit = (Dm > DEF_THR) & br
                    cols = np.argwhere(hit)[:, 2]
                    lv = float((cols < midcol).sum() * vox)
                    rv = float((cols >= midcol).sum() * vox)
                    burden.append([rat, kind, ori, tp, round(lv, 3), round(rv, 3),
                                   round((rv - lv) / max(lv + rv, 1e-9), 3)])
                # 후보: 어느 시점에서든 어두운 덩어리
                for tp, Dm in D.items():
                    other = "PRE" if tp != "PRE" else tag_post
                    cand = (Dm > DEF_THR) & br
                    lab, n = ndimage.label(cand)
                    if n == 0:
                        continue
                    sizes = ndimage.sum(cand, lab, range(1, n + 1))
                    for kk in np.where(sizes >= MIN_VOX)[0]:
                        sel = lab == (kk + 1)
                        pts = np.argwhere(sel)
                        zc, rc, cc = pts.mean(0)
                        xy = pts[:, 1:].astype(float) * np.array([sp[1], sp[0]])
                        ev = np.sort(np.linalg.eigvalsh(np.cov(xy.T))) if len(xy) > 2 else np.ones(2)
                        elong = float(np.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9)))
                        rows.append(dict(
                            rat=rat, kind=kind, ori=ori, found_in=tp,
                            vol_mm3=float(sizes[kk] * vox), n_slice=len(set(pts[:, 0].tolist())),
                            z=round(float(zc), 1), row=int(round(rc)), col=int(round(cc)),
                            def_found=float(Dm[sel].mean()), def_max=float(Dm[sel].max()),
                            def_other=float(D[other][sel].mean()), other=other,
                            elong=elong, depth_mm=float(depth[sel].mean()),
                            mid_mm=abs(cc - midcol) * float(sp[0]),
                            side="L" if cc < midcol else "R"))
                # 몽타주: PRE 결손 | POST 결손
                nz = pre.shape[0]
                zs = [z for z in range(nz) if br[z].sum() > 200]
                fig, axes = plt.subplots(3, len(zs), figsize=(1.7 * len(zs), 5.6), squeeze=False)
                wl = np.percentile(pre[br0], [1, 99.5])
                for c_, z in enumerate(zs):
                    axes[0, c_].imshow(pre[z], cmap="gray", vmin=wl[0], vmax=wl[1])
                    axes[0, c_].set_title(f"z={z}", fontsize=7)
                    for r_, tp in ((1, "PRE"), (2, tag_post)):
                        axes[r_, c_].imshow(np.where(br[z], D[tp][z], np.nan), cmap="magma",
                                            vmin=0, vmax=0.5)
                for a in axes.ravel():
                    a.set_xticks([])
                    a.set_yticks([])
                axes[0, 0].set_ylabel("PRE", fontsize=8)
                axes[1, 0].set_ylabel("결손 PRE", fontsize=8)
                axes[2, 0].set_ylabel(f"결손 {tag_post}", fontsize=8)
                fig.suptitle(f"RAT {rat} {kind} {ori} — 국소 저신호 결손 (1 − S/국소배경, 밝을수록 어두운 점)",
                             fontsize=10)
                fig.tight_layout()
                fig.savefig(os.path.join(out, f"deficit_{kind}_{ori}.png"), dpi=110,
                            bbox_inches="tight")
                plt.close(fig)

    # 같은 덩어리가 PRE·POST 양쪽에서 잡히면 두 줄이 된다 — found_in 으로 구분해 둔다
    rows.sort(key=lambda r: (r["rat"], r["kind"], r["ori"], -r["vol_mm3"] * r["def_found"]))
    with open(os.path.join(base, "hypo_candidates.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        keys = list(rows[0].keys()) if rows else []
        w.writerow(keys)
        for r in rows:
            w.writerow([round(v, 3) if isinstance(v, float) else v for v in r.values()])
    with open(os.path.join(base, "hypo_burden_LR.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["rat", "kind", "orientation", "timepoint", "left_mm3", "right_mm3",
                    "asym_(R-L)/(R+L)"])
        w.writerows(burden)
    return rows, burden


if __name__ == "__main__":
    base = sys.argv[1]
    rows, burden = main(base)
    print(f"후보 {len(rows)}개 → hypo_candidates.csv, 좌우 부담 → hypo_burden_LR.csv")
