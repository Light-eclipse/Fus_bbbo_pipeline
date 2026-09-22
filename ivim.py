# -*- coding: utf-8 -*-
"""260922 IVIM(SE-EPI, b 11개) 분석 — 확산계수 D, 관류분율 f, 의사확산계수 D*.

데이터 구조 (헤더와 신호로 확인)
-------------------------------
* b값마다 별도 시리즈 11개: 10, 20, 40, 80, 120, 200, 250, 400, 600, 800, 1000 s/mm².
  **b=0 시리즈는 없다.** 촬영은 b 내림차순(1000 먼저, 10 마지막)이다.
* 시리즈마다 슬라이스 14장 × 영상 4장. 4장은 DICOM 태그로 구분되지 않고 파일명의
  `imageNNN`만 다르다. 신호로 판별했다:
    - **image1 = b≈0 기준영상.** b와 무관하게 일정하다(폴더1 3060~3075, 폴더4 3004~3045).
    - **image2~4 = 확산 방향 3개.** b가 커질수록 감쇠하고 감쇠 속도가 서로 다르다.
      이방성 조직의 직교 3방향과 일치한다. 방향 벡터는 헤더에 없다.
* SE-EPI, TR 2500, TE 26.22~26.35 ms(b가 클수록 0.13 ms 길다 — T2 48 ms 기준 신호 0.26% 차,
  D로 0.4% 편향, 무시), 64×64, 0.469 mm, 1 mm 14장. GRE 축상과 같은 슬랩을 전치된
  면내 방향으로 찍었다.

정규화
------
시리즈마다 스케일 태그가 다르다(2.88~3.22). 태그 대신 **시리즈 안의 b0 영상으로 정규화**
한다. 태그 차이와 촬영 중 신호 드리프트가 같이 없어진다. 화소별 S0는 11개 b0의 평균,
시리즈별 전역 보정인자 c_b = median(b0_series / S0)만 곱한다.
    S_norm(b, 방향) = S(b, 방향) / (c_b × S0_voxel)
Trace 신호 = 세 방향의 기하평균(trace D = 세 방향 D의 산술평균).

뇌 마스크
---------
IVIM 자체로 문턱을 잡으면 머리 전체(5,555 mm³)가 잡혀 D에 뇌척수액·근육이 섞였다.
그래서 **GRE 축상 PRE로 만든 뇌 마스크(26-09-15에서 검증한 방식, 조직중앙값 0.85배 문턱 +
최대 연결성분)를 DICOM 기하로 IVIM 격자에 옮긴다.** EPI 왜곡·면 어긋남은 GRE를 IVIM 격자로
옮긴 영상과 IVIM b0의 상관을 ±4화소 범위에서 최대화하는 면내 평행이동으로 보정한다.
마스크는 `ivim_mask_check.png`로 반드시 확인할 것.

적합 (분할 IVIM)
----------------
1. b ≥ 200 에서 ln S = ln S_int − b·D (가중 최소제곱, 가중치 S²) → **D**, S_int.
   **f = 1 − S_int** (S_norm(0)=1).
2. D, f 고정, b < 200 (b=0 포함)에서 S = f·exp(−b·D*) + (1−f)·exp(−b·D) 의 D*를
   로그 격자(2~200 ×10⁻³ mm²/s) 전수 탐색.
3. 참고용: 전 구간 단일지수 ADC(관류 섞임), 방향별 D. 방향이 3개라 텐서·FA는 안 된다.

**D*와 f는 화소 단위로 믿지 말 것.** 화소 잡음이 S0의 약 6%인데 f는 0.03 수준이라 관류
성분이 화소 잡음 아래에 있다. 화소 적합에서 D*가 탐색 격자 끝에 붙는 비율이 절반을 넘었다.
그래서 **ROI 평균 신호에 한 번 적합**하고 화소를 재표본(부트스트랩 300회)해 95% 구간을
낸다. 화소 맵은 D만 정량으로 쓰고 f·D*는 공간 분포 참고용이다.

병변 비교
---------
`lesion_summary.csv`의 병변 중심(환자 좌표)을 IVIM 격자로 옮겨 반경 1 mm 원판(가장 가까운
슬라이스) ROI를 잡는다. 대조는 두 가지다.
* 주변 고리: 같은 슬라이스에서 병변 ROI 바깥 1~2 mm 띠. 대칭 가정이 없다.
* 대측 거울: 뇌 마스크 좌우 중심(환자 x) 기준으로 반사한 같은 크기 원판.
다표적 실험이라 대측이 다른 표적일 수 있으므로 두 대조를 같이 본다.

한계
----
* **단일 시점이다.** FUS 전 IVIM이 없어 "FUS로 인한 변화"는 볼 수 없다. 병변 대 대조 비교만 된다.
* 촬영 순서상 IVIM은 PRE 뒤, Gd 전이다(POST1보다 약 10~13분 앞). FUS가 그 사이
  공백(PRE 후 약 34분)에 들어갔는지는 헤더로 확인되지 않는다.
* 64×64라 병변 하나가 화소 몇~십몇 개다. 부분체적이 크다.

실행: python "_ivim_0922.py"        # 폴더 1, 4 전부 (DCE 결과 npz 필요)
"""
import os
import re
import sys
import csv
import glob
import numpy as np
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
plt.rcParams["axes.unicode_minus"] = False       # 맑은 고딕에 U+2212 없음

SCALE_TAG = (0x00E1, 0x1001)
BVALS = [10, 20, 40, 80, 120, 200, 250, 400, 600, 800, 1000]
B_SPLIT = 200
DSTAR_GRID = np.geomspace(2e-3, 200e-3, 120)
NOISE_K = 3.0
DARK_FRAC = 0.85
SHIFT_MAX = 4
ROI_R_MM = 1.0
SHELL_MM = (1.0, 2.0)
N_BOOT = 300
RNG = np.random.default_rng(20260922)
RATS = sys.argv[1:] or ["1", "4"]


# ------------------------------------------------------------ 기하
def dicom_geom(folder):
    fs = glob.glob(os.path.join(folder, "*.dcm"))
    ds = [pydicom.dcmread(f, stop_before_pixels=True) for f in fs]
    iop = np.array([float(x) for x in ds[0].ImageOrientationPatient])
    r, c = iop[:3], iop[3:]
    n = np.cross(r, c)
    ipps = sorted({tuple(float(v) for v in x.ImagePositionPatient) for x in ds},
                  key=lambda p: float(np.dot(np.array(p), n)))
    return dict(ipp0=np.array(ipps[0]), r=r, c=c, n=n,
                ps=float(ds[0].PixelSpacing[0]),
                sz=float(getattr(ds[0], "SpacingBetweenSlices", 1.0) or 1.0))


def vox_to_pat(g, k, i, j):
    return (g["ipp0"][None, :] + np.asarray(j)[:, None] * g["ps"] * g["r"][None, :]
            + np.asarray(i)[:, None] * g["ps"] * g["c"][None, :]
            + np.asarray(k)[:, None] * g["sz"] * g["n"][None, :])


def pat_to_vox(g, P):
    d = P - g["ipp0"][None, :]
    return d @ g["n"] / g["sz"], d @ g["c"] / g["ps"], d @ g["r"] / g["ps"]


def load_series(rat, b):
    d = glob.glob(os.path.join(DATA_ROOT, rat, f"*IVIM*_B{b}_*"))[0]
    fs = glob.glob(os.path.join(d, "*.dcm"))
    h = pydicom.dcmread(fs[0], stop_before_pixels=True)
    tag = float(h[SCALE_TAG].value)
    g = dicom_geom(d)
    vol = np.zeros((4, 14, int(h.Rows), int(h.Columns)), np.float32)
    proj = {}
    for f in fs:
        m = re.search(r"slice(\d+)image(\d+)", os.path.basename(f))
        s, i = int(m.group(1)) - 1, int(m.group(2)) - 1
        ds = pydicom.dcmread(f)
        vol[i, s] = ds.pixel_array.astype(np.float32)
        proj[s] = float(np.dot([float(v) for v in ds.ImagePositionPatient], g["n"]))
    order = np.argsort([proj[s] for s in range(14)])
    vol = vol[:, order]
    return dict(vol=vol / tag, tag=tag, te=float(h.EchoTime), t=str(h.ContentTime), g=g)


# ------------------------------------------------------------ GRE 뇌 마스크 → IVIM
def gre_brain_on_ivim(rat, gi, S0):
    """GRE 축상 PRE 뇌 마스크를 IVIM 격자로 옮기고 면내 어긋남을 상관으로 보정한다."""
    npz = os.path.join(BASE, f"RAT {rat}", "corrected_volumes.npz")
    d = np.load(npz)
    pre = d["gre_axial_PRE"].astype(float)
    tis = d["gre_axial_analysis_mask"]
    med = float(np.median(pre[tis]))
    bright = ndimage.binary_opening(tis & (pre >= DARK_FRAC * med), np.ones((1, 3, 3)))
    lab, n = ndimage.label(bright)
    sizes = ndimage.sum(bright, lab, range(1, n + 1))
    br = lab == (int(np.argmax(sizes)) + 1)
    br = np.stack([ndimage.binary_fill_holes(br[z]) for z in range(br.shape[0])])
    folder = glob.glob(os.path.join(DATA_ROOT, rat, "*GRE_AX_PRE*"))[0]
    gg = dicom_geom(folder)
    K, I, J = np.meshgrid(np.arange(S0.shape[0]), np.arange(S0.shape[1]),
                          np.arange(S0.shape[2]), indexing="ij")
    P = vox_to_pat(gi, K.ravel(), I.ravel(), J.ravel())
    kk, ii, jj = pat_to_vox(gg, P)
    coords = np.vstack([kk, ii, jj])
    # 뇌 마스크는 부분체적 비율로 받아 0.5 문턱 (IVIM 화소 1개 = GRE 화소 약 16개)
    frac = ndimage.map_coordinates(ndimage.uniform_filter(br.astype(np.float32), (1, 4, 4)),
                                   coords, order=1, mode="constant", cval=0).reshape(S0.shape)
    gimg = ndimage.map_coordinates(ndimage.uniform_filter(pre, (1, 4, 4)), coords,
                                   order=1, mode="constant", cval=0).reshape(S0.shape)
    # 면내 평행이동 탐색 (EPI 왜곡·위치 차이)
    ok = gimg > 0
    best = (0, 0, -1.0)
    for di in range(-SHIFT_MAX, SHIFT_MAX + 1):
        for dj in range(-SHIFT_MAX, SHIFT_MAX + 1):
            sh = ndimage.shift(gimg, (0, di, dj), order=1, mode="constant", cval=0)
            k = ok & (sh > 0)
            r = float(np.corrcoef(sh[k], S0[k])[0, 1])
            if r > best[2]:
                best = (di, dj, r)
    r0 = float(np.corrcoef(gimg[ok], S0[ok])[0, 1])
    frac = ndimage.shift(frac, (0, best[0], best[1]), order=1, mode="constant", cval=0)
    gimg = ndimage.shift(gimg, (0, best[0], best[1]), order=1, mode="constant", cval=0)
    mask = frac >= 0.5
    return mask, gimg, dict(shift_rows=best[0], shift_cols=best[1], corr_before=r0,
                            corr_after=best[2], gre_brain_mm3=float(br.sum() * abs(
                                np.prod(d["gre_axial_spacing"]))))


# ------------------------------------------------------------ 적합
def wls_line(b, logS, w):
    sw = w.sum(0) + 1e-12
    bm = (w * b[:, None]).sum(0) / sw
    ym = (w * logS).sum(0) / sw
    cov = (w * (b[:, None] - bm) * (logS - ym)).sum(0)
    var = (w * (b[:, None] - bm) ** 2).sum(0) + 1e-12
    s = cov / var
    return s, ym - s * bm


def fit_mono(b, S, thr):
    good = S > thr
    L = np.where(good, np.log(np.clip(S, 1e-6, None)), 0.0)
    w = np.where(good, S, 0.0) ** 2
    s, ic = wls_line(np.asarray(b, float), L, w)
    return -s, np.exp(ic)


def fit_dstar(b, S, f, D):
    b = np.asarray(b, float)[:, None, None]
    g = DSTAR_GRID[None, :, None]
    model = f[None, None, :] * np.exp(-b * g) + (1 - f)[None, None, :] * np.exp(-b * D[None, None, :])
    k = np.argmin(((model - S[:, None, :]) ** 2).sum(0), axis=0)
    return DSTAR_GRID[k], (k == 0) | (k == len(DSTAR_GRID) - 1)


def ivim_on_curve(bb, curve, thr):
    """평균 곡선 하나에 분할 IVIM 적합. curve:(nb,) (b=0 제외 정규화 신호)."""
    hi = bb >= B_SPLIT
    D, Si = fit_mono(bb[hi], curve[hi][:, None], thr)
    f = np.clip(1 - Si, 0, 1)
    lo = ~hi
    ds, edge = fit_dstar(np.concatenate([[0.0], bb[lo]]),
                         np.concatenate([[1.0], curve[lo]])[:, None], f, D)
    return float(D[0]), float(f[0]), float(ds[0]), bool(edge[0])


def roi_fit(bb, Str, sel, thr):
    """Str:(nb, nvox_brain) 에서 sel(부울, nvox_brain) ROI 평균 적합 + 부트스트랩 95% 구간."""
    ix = np.where(sel)[0]
    est = ivim_on_curve(bb, Str[:, ix].mean(1), thr)
    boots = []
    for _ in range(N_BOOT):
        s = RNG.choice(ix, ix.size, replace=True)
        boots.append(ivim_on_curve(bb, Str[:, s].mean(1), thr)[:3])
    boots = np.array(boots)
    lo, hi = np.percentile(boots, 2.5, 0), np.percentile(boots, 97.5, 0)
    return dict(n=int(ix.size), D=est[0], f=est[1], Dstar=est[2], Dstar_edge=est[3],
                D_lo=lo[0], D_hi=hi[0], f_lo=lo[1], f_hi=hi[1], Ds_lo=lo[2], Ds_hi=hi[2])


# ------------------------------------------------------------ 본체
def main():
    summary, les_rows = [], []
    for rat in RATS:
        out = os.path.join(BASE, f"RAT {rat}", "ivim")
        os.makedirs(out, exist_ok=True)
        print("=" * 80)
        print(f"폴더 {rat}")
        ser = {b: load_series(rat, b) for b in BVALS}
        gi = ser[10]["g"]
        S0 = np.stack([ser[b]["vol"][0] for b in BVALS]).mean(0)
        mask, gimg, reg = gre_brain_on_ivim(rat, gi, S0)
        vox_mm3 = gi["ps"] ** 2 * gi["sz"]
        print(f"  GRE→IVIM 면내 어긋남 보정 (행 {reg['shift_rows']:+d}, 열 {reg['shift_cols']:+d}) "
              f"화소, 상관 {reg['corr_before']:.3f} → {reg['corr_after']:.3f}")
        print(f"  뇌 마스크 {int(mask.sum())}화소 = {mask.sum()*vox_mm3:.0f} mm³ "
              f"(GRE 원본 {reg['gre_brain_mm3']:.0f} mm³)")

        cb = {b: float(np.median(ser[b]["vol"][0][mask] / (S0[mask] + 1e-6))) for b in BVALS}
        t_order = sorted(BVALS, key=lambda b: ser[b]["t"])
        drift = [(cb[b] - 1) * 100 for b in t_order]
        print(f"  b0 드리프트(촬영순) {drift[0]:+.2f}% → {drift[-1]:+.2f}% "
              f"(범위 {max(drift)-min(drift):.2f}%p)")
        bg = S0 < np.percentile(S0, 10)
        sigma = float(np.mean([ser[b]["vol"][1:][:, bg].mean() for b in BVALS])) / 1.253
        sig_n = sigma / float(np.median(S0[mask]))
        thr = NOISE_K * sig_n
        print(f"  잡음 σ = S0 뇌 중앙값의 {sig_n*100:.2f}% → 적합 문턱 {thr:.3f}")

        idx = np.where(mask)
        nv = idx[0].size
        Sdir = np.zeros((len(BVALS), 3, nv))
        for k, b in enumerate(BVALS):
            den = cb[b] * S0[idx] + 1e-6
            for j in range(3):
                Sdir[k, j] = ser[b]["vol"][j + 1][idx] / den
        Str = np.exp(np.log(np.clip(Sdir, 1e-6, None)).mean(1))
        bb = np.array(BVALS, float)
        hi = bb >= B_SPLIT

        # 화소 맵
        D, Si = fit_mono(bb[hi], Str[hi], thr)
        f = np.clip(1 - Si, 0, 1)
        Dst, edge = fit_dstar(np.concatenate([[0.0], bb[~hi]]),
                              np.concatenate([np.ones((1, nv)), Str[~hi]], 0), f, D)
        ADC, _ = fit_mono(np.concatenate([[0.0], bb]), np.concatenate([np.ones((1, nv)), Str], 0), thr)
        Ddir = np.stack([fit_mono(bb[hi], Sdir[hi, j], thr)[0] for j in range(3)])

        def vol_of(x):
            a = np.full(mask.shape, np.nan, np.float32)
            a[idx] = x
            return a
        maps = dict(D=vol_of(D), f=vol_of(f), Dstar=vol_of(Dst), ADC=vol_of(ADC),
                    D_dir1=vol_of(Ddir[0]), D_dir2=vol_of(Ddir[1]), D_dir3=vol_of(Ddir[2]),
                    dstar_at_grid_edge=vol_of(edge.astype(float)), S0=S0.astype(np.float32),
                    gre_pre_on_ivim=gimg.astype(np.float32), mask=mask)
        np.savez_compressed(os.path.join(out, "ivim_maps.npz"), **maps, bvals=np.array(BVALS),
                            ipp0=gi["ipp0"], row_dir=gi["r"], col_dir=gi["c"], slice_dir=gi["n"],
                            pixel_mm=gi["ps"], slice_mm=gi["sz"], sigma_norm=sig_n)

        # 뇌 전체 ROI 적합
        wb = roi_fit(bb, Str, np.ones(nv, bool), thr)
        print(f"  [뇌 전체 ROI] D {wb['D']*1e3:.3f} ({wb['D_lo']*1e3:.3f}~{wb['D_hi']*1e3:.3f}) ×10⁻³ mm²/s, "
              f"f {wb['f']:.4f} ({wb['f_lo']:.4f}~{wb['f_hi']:.4f}), "
              f"D* {wb['Dstar']*1e3:.2f} ({wb['Ds_lo']*1e3:.2f}~{wb['Ds_hi']*1e3:.2f}) ×10⁻³"
              f"{'  [D* 격자 끝]' if wb['Dstar_edge'] else ''}")
        print(f"  [화소 분포] D 중앙값 {np.median(D)*1e3:.3f} (IQR {np.percentile(D,25)*1e3:.3f}~"
              f"{np.percentile(D,75)*1e3:.3f}), 화소 D* 격자 끝 {edge.mean()*100:.1f}%")
        print(f"  방향별 D 중앙값 {' / '.join(f'{np.median(Ddir[j])*1e3:.3f}' for j in range(3))} ×10⁻³")
        summary.append(dict(rat=rat, brain_vox=nv, brain_mm3=round(nv * vox_mm3, 1),
                            shift_rows=reg["shift_rows"], shift_cols=reg["shift_cols"],
                            reg_corr=round(reg["corr_after"], 3), sigma_pct=round(sig_n * 100, 2),
                            drift_range_pp=round(max(drift) - min(drift), 2),
                            D=wb["D"] * 1e3, D_lo=wb["D_lo"] * 1e3, D_hi=wb["D_hi"] * 1e3,
                            f=wb["f"], f_lo=wb["f_lo"], f_hi=wb["f_hi"],
                            Dstar=wb["Dstar"] * 1e3, Ds_lo=wb["Ds_lo"] * 1e3, Ds_hi=wb["Ds_hi"] * 1e3,
                            Dstar_edge=wb["Dstar_edge"],
                            voxel_D_median=np.median(D) * 1e3, voxel_Dstar_edge_pct=edge.mean() * 100,
                            ADC_median=np.median(ADC) * 1e3,
                            Ddir_median=" / ".join(f"{np.median(Ddir[j])*1e3:.3f}" for j in range(3))))

        # ---- 병변 비교
        # 다중획득 재현(신뢰 높음) 병변만 비교한다. 기본 문턱(15%p)과 감도 분석(10%p) 목록을
        # 합치고, 2 mm 안에서 겹치면 하나로 본다. 출처 문턱을 같이 적는다.
        cand = []
        for fn, src in (("lesion_summary.csv", "15%p"), ("lesion_summary_pp10.csv", "10%p")):
            fp = os.path.join(BASE, fn)
            if not os.path.exists(fp):
                continue
            for x in csv.DictReader(open(fp, encoding="utf-8-sig")):
                if x["rat_folder"] != rat or not x["confidence"].startswith("높음"):
                    continue
                P_ = np.array([float(v) for v in x["patient_xyz_mm"].split()])
                if any(np.linalg.norm(P_ - c_[0]) <= 2.0 for c_ in cand):
                    continue
                x = dict(x)
                x["lesion"] = f"{src}#{x['lesion']}"
                cand.append((P_, x))
        if cand:
            flat = np.full(mask.shape, -1, int)
            flat[idx] = np.arange(nv)
            K, I, J = np.meshgrid(np.arange(14), np.arange(S0.shape[1]), np.arange(S0.shape[2]),
                                  indexing="ij")
            Pall = vox_to_pat(gi, K.ravel(), I.ravel(), J.ravel()).reshape(14, S0.shape[1],
                                                                           S0.shape[2], 3)
            # 면내 보정된 마스크와 같은 좌표계를 쓰도록 병변 좌표도 같은 이동을 적용한다
            xmid = float(Pall[mask][:, 0].mean())
            for P, x in cand:
                kz, ki, kj = pat_to_vox(gi, P[None, :])
                kz = int(round(float(kz[0])))
                ci = float(ki[0]) + reg["shift_rows"]
                cj = float(kj[0]) + reg["shift_cols"]
                if not (0 <= kz < 14):
                    continue
                Pm = P.copy()
                Pm[0] = 2 * xmid - P[0]
                mz, mi, mj = pat_to_vox(gi, Pm[None, :])
                mi = float(mi[0]) + reg["shift_rows"]
                mj = float(mj[0]) + reg["shift_cols"]
                yy, xx = np.mgrid[:S0.shape[1], :S0.shape[2]]
                dist_l = np.hypot(yy - ci, xx - cj) * gi["ps"]
                dist_m = np.hypot(yy - mi, xx - mj) * gi["ps"]
                roi = np.zeros(mask.shape, bool)
                roi[kz] = dist_l <= ROI_R_MM
                shell = np.zeros(mask.shape, bool)
                shell[kz] = (dist_l > SHELL_MM[0]) & (dist_l <= SHELL_MM[1])
                mir = np.zeros(mask.shape, bool)
                mir[kz] = dist_m <= ROI_R_MM
                res = {}
                for nm, R in (("lesion", roi), ("shell", shell), ("mirror", mir)):
                    R = R & mask
                    if R.sum() < 3:
                        res[nm] = None
                        continue
                    sel = np.zeros(nv, bool)
                    sel[flat[R]] = True
                    res[nm] = roi_fit(bb, Str, sel, thr)
                if res["lesion"] is None:
                    continue
                print(f"  병변 {x['lesion']} ({x['confidence']}, 깊이 {x['depth_below_brain_surface_mm']} mm, "
                      f"IVIM z={kz})")
                for nm in ("lesion", "shell", "mirror"):
                    r = res[nm]
                    if r is None:
                        print(f"     {nm:6}: 뇌 마스크 안 화소 부족")
                        continue
                    print(f"     {nm:6} n={r['n']:3d}  D {r['D']*1e3:.3f} ({r['D_lo']*1e3:.3f}~"
                          f"{r['D_hi']*1e3:.3f})  f {r['f']:.3f} ({r['f_lo']:.3f}~{r['f_hi']:.3f})  "
                          f"D* {r['Dstar']*1e3:6.2f} ({r['Ds_lo']*1e3:.2f}~{r['Ds_hi']*1e3:.2f})"
                          f"{' [격자끝]' if r['Dstar_edge'] else ''}")
                    les_rows.append([rat, x["lesion"], x["confidence"], x["depth_below_brain_surface_mm"],
                                     kz, nm, r["n"], round(r["D"] * 1e3, 4), round(r["D_lo"] * 1e3, 4),
                                     round(r["D_hi"] * 1e3, 4), round(r["f"], 4), round(r["f_lo"], 4),
                                     round(r["f_hi"], 4), round(r["Dstar"] * 1e3, 3),
                                     round(r["Ds_lo"] * 1e3, 3), round(r["Ds_hi"] * 1e3, 3),
                                     "예" if r["Dstar_edge"] else "아니오"])

        # ---- 그림: 뇌 평균 감쇠 곡선
        mS = np.concatenate([[1.0], Str.mean(1)])
        mSd = [np.concatenate([[1.0], Sdir[:, j].mean(1)]) for j in range(3)]
        b_all = np.concatenate([[0.0], bb])
        xb = np.linspace(0, 1000, 300)
        Dm, fm, Dsm = wb["D"], wb["f"], wb["Dstar"]
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        for a, ylog in zip(ax, (False, True)):
            for j, col in enumerate(("tab:blue", "tab:green", "tab:purple")):
                a.plot(b_all, mSd[j], "o", ms=4, color=col, alpha=0.6, label=f"방향 {j+1}")
            a.plot(b_all, mS, "ks", ms=5, label="trace(기하평균)")
            a.plot(xb, fm * np.exp(-xb * Dsm) + (1 - fm) * np.exp(-xb * Dm), "r-",
                   label=f"ROI 적합 D={Dm*1e3:.3f}, f={fm:.3f}, D*={Dsm*1e3:.1f}")
            a.plot(xb, (1 - fm) * np.exp(-xb * Dm), "r:", lw=1, label="조직 성분만")
            a.axvline(B_SPLIT, color="gray", ls="--", lw=0.8)
            if ylog:
                a.set_yscale("log")
            a.set_xlabel("b (s/mm²)")
            a.set_ylabel("S / S0")
            a.grid(alpha=0.3)
        ax[0].legend(fontsize=7)
        fig.suptitle(f"폴더 {rat} IVIM — 뇌 평균 신호 (회색 점선 = D 적합 경계 b={B_SPLIT})", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "ivim_decay_curve.png"), dpi=130)
        plt.close(fig)

        # ---- 그림: 맵 (GRE 축상과 같은 방향으로 보이게 전치)
        def show(v):
            return np.transpose(v, (0, 2, 1))[:, :, ::-1]
        panels = [("GRE PRE→IVIM", gimg, None, None, "gray"),
                  ("IVIM 평균 b0", S0, None, None, "gray"),
                  ("D (×10⁻³ mm²/s)", maps["D"] * 1e3, 0.3, 1.2, "viridis"),
                  ("f (참고)", maps["f"], 0, 0.2, "magma"),
                  ("ADC 전구간 (×10⁻³)", maps["ADC"] * 1e3, 0.3, 1.2, "viridis")]
        zs = [z for z in range(14) if mask[z].sum() > 20]
        fig, axes = plt.subplots(len(panels), len(zs), figsize=(1.5 * len(zs), 1.65 * len(panels) + 0.6),
                                 squeeze=False)
        Ms = show(mask.astype(float))
        for r_, (nm, v, lo_, hi_, cm) in enumerate(panels):
            V = show(v)
            if lo_ is None:
                lo_, hi_ = np.percentile(V[V > 0], [2, 99]) if (V > 0).any() else (0, 1)
            im = None
            for c_, z in enumerate(zs):
                im = axes[r_, c_].imshow(V[z], cmap=cm, vmin=lo_, vmax=hi_)
                if r_ < 2:
                    axes[r_, c_].contour(Ms[z], levels=[0.5], colors="lime", linewidths=0.6)
                axes[r_, c_].set_xticks([])
                axes[r_, c_].set_yticks([])
                if r_ == 0:
                    axes[r_, c_].set_title(f"z={z}", fontsize=7)
            axes[r_, 0].set_ylabel(nm, fontsize=7)
            if r_ >= 2:
                fig.colorbar(im, ax=axes[r_, -1], fraction=0.08, pad=0.04)
        fig.suptitle(f"폴더 {rat} IVIM — 윗 두 줄 초록 = GRE에서 옮긴 뇌 마스크", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "ivim_maps_montage.png"), dpi=120, bbox_inches="tight")
        plt.close(fig)

        with open(os.path.join(out, "ivim_series.csv"), "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["b", "content_time", "TE_ms", "scale_tag", "b0_drift_factor",
                        "trace_mean_norm", "dir1_mean", "dir2_mean", "dir3_mean"])
            for k, b in enumerate(BVALS):
                w.writerow([b, ser[b]["t"], ser[b]["te"], round(ser[b]["tag"], 4), round(cb[b], 5),
                            round(float(Str[k].mean()), 5), *[round(float(Sdir[k, j].mean()), 5)
                                                               for j in range(3)]])

    with open(os.path.join(BASE, "ivim_summary.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(list(summary[0].keys()))
        for r in summary:
            w.writerow([round(v, 5) if isinstance(v, float) else v for v in r.values()])
    if les_rows:
        with open(os.path.join(BASE, "ivim_lesion_compare.csv"), "w", newline="",
                  encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["rat_folder", "lesion", "confidence", "depth_mm", "ivim_slice", "roi",
                        "n_vox", "D_1e3", "D_lo", "D_hi", "f", "f_lo", "f_hi",
                        "Dstar_1e3", "Ds_lo", "Ds_hi", "Dstar_at_grid_edge"])
            w.writerows(les_rows)
    print("\n저장: RAT */ivim/, ivim_summary.csv" + (", ivim_lesion_compare.csv" if les_rows else ""))


if __name__ == "__main__":
    main()
