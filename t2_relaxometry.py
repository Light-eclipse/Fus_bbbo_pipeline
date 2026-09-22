# -*- coding: utf-8 -*-
"""260922 MREPT(8에코 NIfTI) → T2 맵.

데이터
------
* `0922_MREPT-NO?_20260922_01.nii/vol_image001echoNNN.nii` 8개. 128×128×14, float32,
  scl_slope 100(26-08-20에서 확인한 "nii = 참값 × 100"과 같은 규칙).
* `recon.nii`는 **MREPT와 파일 단위로 동일**하다(8에코 전부 md5 일치). 위상이 없으므로
  **EPT는 계산할 수 없다.**
* NIfTI 기하가 쓸 수 없다. qform 쿼터니언이 비정규라 nibabel 기본 로드가 실패하고
  sform은 0이다. 그래서 GRE 축상 PRE(128로 축소)와의 상관으로 방향을 찾았다:
  배열 (x,y,z) → (z,y,x) 전치 후 y 반전이 GRE 축상과 맞는다(상관 0.87 / 0.86).
  남은 면내 어긋남은 ±4화소 상관 탐색으로 보정한다.
* **에코 시간이 어디에도 없다.** DICOM이 없고 NIfTI에도 TE가 없다. 26-08-20 MREPT
  (mems, 10에코, 첫 TE 7.0 ms)와 같은 이름이라 **ΔTE = 7.0 ms로 가정**한다. 그러나
  이번은 8에코라 같은 프로토콜이라는 보장이 없다. ms 단위 T2는 이 상수에 선형 비례하고,
  함께 저장하는 **에코당 감쇠율 R2_per_echo는 가정과 무관**하다.
* 촬영 시각도 없다(파일 시각은 내보낸 시각). Gd 전인지 후인지 모른다. Gd 후라면 T2가
  Gd의 T2 단축만큼 짧게 나온다.

짝·홀 에코 진동
---------------
에코별 평균이 매끄럽게 줄지 않는다. 연속 에코 비가 0.889, 0.713, 0.856, 0.759, 0.866,
0.791, 0.882로 **짝수 에코가 추세보다 밝다**. 다중에코 스핀에코(CPMG)에서 재초점 펄스가
정확히 180°가 아닐 때 자극 에코 경로가 짝수 에코에 더해지는 전형적 양상이다.
단일지수 적합이 이 진동에 휘둘리므로 세 가지를 따로 낸다: 전 에코, 홀수 에코(1,3,5,7),
짝수 에코(2,4,6,8). 홀·짝은 에코 간격이 2ΔTE가 된다. 세 값의 차이가 이 방법의 불확실성이다.
자극 에코를 제대로 다루려면 EPG 모형 적합이 필요한데, 플립각 정보가 없어 하지 않았다.

실행: python "_t2_0922.py"
"""
import os
import sys
import csv
import glob
import numpy as np
import nibabel as nib
import pydicom
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))
DATA_ROOT = os.path.join(os.path.dirname(BASE), "260922_BBBO_200G")
sys.path.insert(0, os.path.join(ROOT, "MRI 분석 코드"))
from mri_dce import _set_korean_font
_set_korean_font()
plt.rcParams["axes.unicode_minus"] = False

DTE_MS = 7.0            # 가정 (26-08-20 MREPT 첫 TE)
DARK_FRAC = 0.85
NOISE_K = 3.0
SHIFT_MAX = 4
RATS = sys.argv[1:] or ["1", "4"]


def read_nii(f):
    with open(f, "rb") as fh:
        h = nib.Nifti1Header.from_fileobj(fh)
        return np.asarray(h.data_from_fileobj(fh), np.float64)


def to_gre(a):
    """MREPT (x,y,z) → GRE 축상 배열 방향 (z, row, col)."""
    return np.transpose(a, (2, 1, 0))[:, ::-1, :]


def gre_brain_128(rat):
    d = np.load(os.path.join(BASE, f"RAT {rat}", "corrected_volumes.npz"))
    pre = d["gre_axial_PRE"].astype(float)
    tis = d["gre_axial_analysis_mask"]
    med = float(np.median(pre[tis]))
    bright = ndimage.binary_opening(tis & (pre >= DARK_FRAC * med), np.ones((1, 3, 3)))
    lab, n = ndimage.label(bright)
    sizes = ndimage.sum(bright, lab, range(1, n + 1))
    br = lab == (int(np.argmax(sizes)) + 1)
    br = np.stack([ndimage.binary_fill_holes(br[z]) for z in range(br.shape[0])])
    # 2x2 블록 평균으로 256 → 128
    f = br.reshape(br.shape[0], 128, 2, 128, 2).mean((2, 4))
    g = pre.reshape(pre.shape[0], 128, 2, 128, 2).mean((2, 4))
    return f, g, float(np.prod(np.abs(d["gre_axial_spacing"])))


def fit(E, dte, sigma):
    """E:(ne, nvox). 가중 로그선형. 반환 R2(에코당), T2(ms), S0, 유효점수."""
    ne = E.shape[0]
    n = np.arange(1, ne + 1, dtype=float)
    good = E > NOISE_K * sigma
    L = np.where(good, np.log(np.clip(E, 1e-6, None)), 0.0)
    w = np.where(good, E, 0.0) ** 2
    sw = w.sum(0) + 1e-12
    nm = (w * n[:, None]).sum(0) / sw
    lm = (w * L).sum(0) / sw
    s = (w * (n[:, None] - nm) * (L - lm)).sum(0) / ((w * (n[:, None] - nm) ** 2).sum(0) + 1e-12)
    R2 = -s
    T2 = np.where(R2 > 0, dte / R2, np.nan)
    return R2, T2, np.exp(lm - s * nm), good.sum(0)


def main():
    rows = []
    for rat in RATS:
        out = os.path.join(BASE, f"RAT {rat}", "t2")
        os.makedirs(out, exist_ok=True)
        sub = glob.glob(os.path.join(DATA_ROOT, rat, "0922_MREPT*.nii"))[0]
        fs = sorted(glob.glob(os.path.join(sub, "vol_*.nii")))
        E = np.stack([to_gre(read_nii(f)) for f in fs])            # (8,14,128,128)
        frac, g128, _ = gre_brain_128(rat)
        # 면내 어긋남 보정
        e1 = E[0]
        ok = g128 > 0
        best = (0, 0, -1.0)
        for di in range(-SHIFT_MAX, SHIFT_MAX + 1):
            for dj in range(-SHIFT_MAX, SHIFT_MAX + 1):
                sh = ndimage.shift(g128, (0, di, dj), order=1, mode="constant", cval=0)
                k = ok & (sh > 0)
                r = float(np.corrcoef(sh[k], e1[k])[0, 1])
                if r > best[2]:
                    best = (di, dj, r)
        frac = ndimage.shift(frac, (0, best[0], best[1]), order=1, mode="constant", cval=0)
        mask = frac >= 0.5
        vox = 0.234375 ** 2 * 1.0
        bg = e1 < np.percentile(e1, 10)
        sigma = float(np.mean([E[k][bg].mean() for k in range(E.shape[0])])) / 1.253
        idx = np.where(mask)
        X = E[:, idx[0], idx[1], idx[2]]
        print("=" * 76)
        print(f"폴더 {rat}: GRE→MREPT 면내 보정 ({best[0]:+d}, {best[1]:+d}) 화소, 상관 {best[2]:.3f}, "
              f"뇌 마스크 {int(mask.sum())}화소 = {mask.sum()*vox:.0f} mm³")
        res = {}
        for nm, sl, step in (("전 에코", slice(0, 8), 1), ("홀수 에코", slice(0, 8, 2), 2),
                             ("짝수 에코", slice(1, 8, 2), 2)):
            R2, T2, S0, ng = fit(X[sl], DTE_MS * step, sigma)
            v = T2[np.isfinite(T2) & (ng >= 3)]
            res[nm] = (R2 / step, T2)
            q1, md, q3 = np.percentile(v, [25, 50, 75])
            print(f"  {nm:6}: T2 중앙값 {md:6.2f} ms (IQR {q1:.2f}~{q3:.2f}), "
                  f"에코당 R2 중앙값 {np.median(R2/step):.4f}, 유효 화소 {v.size}")
            rows.append([rat, nm, round(md, 2), round(q1, 2), round(q3, 2),
                         round(float(np.median(R2 / step)), 5), v.size, DTE_MS])
        # 에코별 뇌 평균
        mE = X.mean(1)
        print("  에코별 뇌 평균 비(연속):", [round(float(mE[k + 1] / mE[k]), 3) for k in range(7)])

        def vol_of(x):
            a = np.full(mask.shape, np.nan, np.float32)
            a[idx] = x
            return a
        np.savez_compressed(os.path.join(out, "t2_maps.npz"),
                            T2_all_ms=vol_of(res["전 에코"][1]), T2_odd_ms=vol_of(res["홀수 에코"][1]),
                            T2_even_ms=vol_of(res["짝수 에코"][1]),
                            R2_per_echo_all=vol_of(res["전 에코"][0]),
                            echoes=E.astype(np.float32), mask=mask, dte_ms_assumed=DTE_MS,
                            shift=np.array(best[:2]))

        # 그림: 에코 감쇠
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        te = DTE_MS * np.arange(1, 9)
        ax.semilogy(te, mE, "ko-", label="뇌 평균 (전 에코)")
        ax.semilogy(te[0::2], mE[0::2], "s", color="tab:blue", ms=8, mfc="none", label="홀수 에코")
        ax.semilogy(te[1::2], mE[1::2], "^", color="tab:red", ms=8, mfc="none", label="짝수 에코")
        ax.set_xlabel(f"TE (ms, ΔTE={DTE_MS} 가정)")
        ax.set_ylabel("신호 (로그)")
        ax.set_title(f"폴더 {rat} MREPT 에코 감쇠 — 짝수 에코가 추세보다 밝다", fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "t2_decay.png"), dpi=130)
        plt.close(fig)

        # 그림: 맵
        zs = [z for z in range(14) if mask[z].sum() > 100]
        panels = [("에코 1", E[0], None), ("T2 전 에코 (ms)", vol_of(res["전 에코"][1]), (20, 80)),
                  ("T2 홀수 (ms)", vol_of(res["홀수 에코"][1]), (20, 80)),
                  ("T2 짝수 (ms)", vol_of(res["짝수 에코"][1]), (20, 80))]
        fig, axes = plt.subplots(len(panels), len(zs), figsize=(1.5 * len(zs), 1.65 * len(panels) + 0.5),
                                 squeeze=False)
        for r_, (nm, v, lim) in enumerate(panels):
            if lim is None:
                lim = tuple(np.percentile(v[v > 0], [2, 99]))
            im = None
            for c_, z in enumerate(zs):
                im = axes[r_, c_].imshow(v[z], cmap="gray" if r_ == 0 else "turbo",
                                         vmin=lim[0], vmax=lim[1])
                if r_ == 0:
                    axes[r_, c_].contour(mask[z], levels=[0.5], colors="lime", linewidths=0.5)
                    axes[r_, c_].set_title(f"z={z}", fontsize=7)
                axes[r_, c_].set_xticks([])
                axes[r_, c_].set_yticks([])
            axes[r_, 0].set_ylabel(nm, fontsize=7)
            if r_ > 0:
                fig.colorbar(im, ax=axes[r_, -1], fraction=0.08, pad=0.04)
        fig.suptitle(f"폴더 {rat} T2 맵 (ΔTE {DTE_MS} ms 가정, 초록 = GRE에서 옮긴 뇌 마스크)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "t2_maps_montage.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

    with open(os.path.join(BASE, "t2_summary.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["rat_folder", "echo_set", "T2_median_ms", "T2_q1", "T2_q3",
                    "R2_per_echo_median", "n_vox", "dTE_ms_assumed"])
        w.writerows(rows)
    print("\n저장: RAT */t2/, t2_summary.csv")


if __name__ == "__main__":
    main()
