# -*- coding: utf-8 -*-
"""면외 이동 보정 후에도 남는 신호 감소 덩어리를 환자 좌표로 옮겨, 독립 획득 사이에서 같은
자리에 나오는지 확인하고 그림을 만든다.

네 획득(GRE·FSE × 축상·관상)은 시퀀스·처방이 다르다. 같은 물리 좌표에서 감소가 재현되면
한 시리즈의 정합·보간 오차로 설명하기 어렵다. 9/22는 IVIM EPI b0(TE 26 ms, 자화율에 민감,
Gd 전)도 같은 자리에서 대측과 비교한다. 출혈이면 EPI에서 크게 어두워야 한다.

실행: python negchange_localize.py
"""
import os
import sys
import csv
import glob
import numpy as np
import pydicom
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mri_dce import _set_korean_font
_set_korean_font()
plt.rcParams["axes.unicode_minus"] = False

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = {"26-09-11": "260911_BBBO_80G", "26-09-15": "260915_BBBO_120G",
        "26-09-22": "260922_BBBO_200G"}
CASES = [("26-09-11", "4"), ("26-09-11", "3"), ("26-09-11", "2"),
         ("26-09-15", "1"), ("26-09-15", "3"), ("26-09-15", "4"),
         ("26-09-22", "1"), ("26-09-22", "4")]
SHIFT = {(x["date"], x["rat"], x["kind"], x["orientation"]): float(x["throughplane_shift_slice"])
         for x in csv.DictReader(open(os.path.join(R, "MRI 분석 코드", "negchange_diag_result.csv"),
                                      encoding="utf-8-sig"))}
DARK_FRAC, THR, MIN_MM3, CLUSTER_MM = 0.85, -0.10, 0.5, 3.0


def brain_mask(pre, tis):
    med = float(np.median(pre[tis]))
    bright = ndimage.binary_opening(tis & (pre >= DARK_FRAC * med), np.ones((1, 3, 3)))
    lab, n = ndimage.label(bright)
    sizes = ndimage.sum(bright, lab, range(1, n + 1))
    br = lab == (int(np.argmax(sizes)) + 1)
    return np.stack([ndimage.binary_fill_holes(br[z]) for z in range(br.shape[0])])


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


def v2p(g, z, i, j):
    return g["ipp0"] + j * g["ps"] * g["r"] + i * g["ps"] * g["c"] + z * g["sz"] * g["n"]


def p2v(g, P):
    d = P - g["ipp0"]
    return d @ g["n"] / g["sz"], d @ g["c"] / g["ps"], d @ g["r"] / g["ps"]


summary = []
for date, rat in CASES:
    base = os.path.join(R, date, "2. 분석 결과", f"RAT {rat}")
    d = np.load(os.path.join(base, "corrected_volumes.npz"))
    droot = os.path.join(R, date, DATA[date], rat)
    out = os.path.join(base, "negchange")
    os.makedirs(out, exist_ok=True)
    clusters, maps = [], {}
    for kind in ("gre", "fse"):
        for ori, tag in (("axial", "AX"), ("coronal", "COR")):
            pre = d[f"{kind}_{ori}_PRE"].astype(float)
            posts = sorted([k for k in d.files if k.startswith(f"{kind}_{ori}_POST")],
                           key=lambda k: int(k.rsplit("POST", 1)[1] or 1))
            dz = SHIFT[(date, rat, kind, ori)]
            if abs(dz) >= 1.45:          # 탐색 경계에 붙은 추정은 믿지 않는다
                dz = 0.0
            post = ndimage.shift(d[posts[-1]].astype(float), (dz, 0, 0), order=1, mode="nearest")
            sp = d[f"{kind}_{ori}_spacing"]
            vox = abs(float(np.prod(sp)))
            br0 = brain_mask(pre, d[f"{kind}_{ori}_analysis_mask"])
            br = ndimage.binary_erosion(br0, np.ones((1, 9, 9)))
            nz = pre.shape[0]
            k = int(np.ceil(abs(dz)))
            if dz > 0:
                br[:k] = False
            elif dz < 0:
                br[nz - k:] = False
            ps = ndimage.gaussian_filter(pre, (0, 1, 1))
            qs = ndimage.gaussian_filter(post, (0, 1, 1))
            c = np.where(br, qs / np.maximum(ps, 1e-6) - 1, np.nan)
            maps[(kind, ori)] = (pre, post, c, br0, dz)
            g = geom(glob.glob(os.path.join(droot, f"*{kind.upper()}_{tag}_PRE*"))[0])
            dec = np.nan_to_num(c, nan=0) < THR
            lab, n = ndimage.label(dec)
            if not n:
                continue
            sizes = ndimage.sum(dec, lab, range(1, n + 1)) * vox
            for kk in np.where(sizes >= MIN_MM3)[0]:
                pts = np.argwhere(lab == kk + 1)
                z, i, j = pts.mean(0)
                clusters.append(dict(kind=kind, ori=ori, vol=float(sizes[kk]),
                                     mean=float(np.nanmean(c[lab == kk + 1])) * 100,
                                     P=v2p(g, z, i, j), z=z, i=i, j=j))
    # 물리 좌표로 묶기
    used = [False] * len(clusters)
    groups = []
    for a in sorted(range(len(clusters)), key=lambda t: -clusters[t]["vol"]):
        if used[a]:
            continue
        grp = [a]
        used[a] = True
        for b in range(len(clusters)):
            if not used[b] and np.linalg.norm(clusters[a]["P"] - clusters[b]["P"]) <= CLUSTER_MM:
                grp.append(b)
                used[b] = True
        groups.append(grp)
    print("=" * 90)
    print(f"{date} RAT {rat}: 이동 보정 후 감소 덩어리(≥{MIN_MM3} mm³) {len(clusters)}개 → 위치 {len(groups)}곳")
    rep = []
    for gi, grp in enumerate(groups):
        mem = [clusters[t] for t in grp]
        acq = sorted({f"{m['kind']}-{m['ori'][:3]}" for m in mem})
        oris = {m["ori"] for m in mem}
        P = np.mean([m["P"] for m in mem], 0)
        vol = max(m["vol"] for m in mem)
        mn = float(np.mean([m["mean"] for m in mem]))
        flag = "두 방향" if len(oris) == 2 else "한 방향"
        if len(acq) >= 2:
            rep.append((vol, P, acq, mn, flag))
        summary.append([date, rat, gi, round(vol, 2), round(mn, 1), len(acq), " ".join(acq), flag,
                        " ".join(f"{v:.2f}" for v in P)])
    rep.sort(key=lambda t: -t[0])
    for vol, P, acq, mn, flag in rep[:6]:
        print(f"   {vol:7.2f} mm³  평균 {mn:+.1f}%  획득 {len(acq)} {acq}  [{flag}]  xyz {np.round(P, 2)}")
    # 그림: 두 방향 이상에서 재현된 상위 3곳
    for rank, (vol, P, acq, mn, flag) in enumerate(rep[:3], 1):
        fig, axes = plt.subplots(2, 6, figsize=(21, 7.4))
        for row_, (ori, tag) in enumerate((("axial", "AX"), ("coronal", "COR"))):
            for col, kind in ((0, "gre"), (3, "fse")):
                pre, post, c, br0, dz = maps[(kind, ori)]
                g = geom(glob.glob(os.path.join(droot, f"*{kind.upper()}_{tag}_PRE*"))[0])
                z, i, j = p2v(g, P)
                zi = int(round(z))
                ax = axes[row_, col:col + 3]
                if not 0 <= zi < pre.shape[0]:
                    for a in ax:
                        a.text(0.5, 0.5, "슬랩 밖", ha="center", transform=a.transAxes)
                        a.axis("off")
                    continue
                wl = np.percentile(pre[br0], [1, 99.5])
                for a, img, nm, kw in ((ax[0], pre[zi], "PRE", dict(cmap="gray", vmin=wl[0], vmax=wl[1])),
                                       (ax[1], post[zi], f"POST(면외 {dz:+.1f} 보정)",
                                        dict(cmap="gray", vmin=wl[0], vmax=wl[1])),
                                       (ax[2], c[zi] * 100, "변화 % (파랑=감소)",
                                        dict(cmap="RdBu_r", vmin=-30, vmax=30))):
                    a.imshow(img, **kw)
                    a.add_patch(plt.Circle((j, i), 1.5 / g["ps"], fill=False, ec="lime", lw=1.3))
                    a.set_title(f"{kind.upper()} {ori} z={zi}  {nm}", fontsize=8)
                    a.axis("off")
        fig.suptitle(f"{date} RAT {rat} 감소 위치 #{rank}: 최대 {vol:.2f} mm³, 평균 {mn:+.1f}%, "
                     f"획득 {acq} [{flag}]  (초록 원 반경 1.5 mm)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"residual_decrease_{rank}.png"), dpi=110, bbox_inches="tight")
        plt.close(fig)

with open(os.path.join(R, "MRI 분석 코드", "negchange_localize_result.csv"), "w", newline="",
          encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["date", "rat", "group", "max_vol_mm3", "mean_change_pct", "n_acq", "acquisitions",
                "orientations", "patient_xyz_mm"])
    w.writerows(summary)
