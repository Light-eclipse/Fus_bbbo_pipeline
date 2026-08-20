# -*- coding: utf-8 -*-
"""260820 BBB RAT EPT — 개체 1·2 통합 분석 드라이버.

세 갈래를 돌린다.
  A. DCE   : GRE AX/COR, 시리즈별 스케일 보정 후 Gd 증강(기준 = POST, 즉 FUS 후·Gd 전)
  B. 이완  : 다중에코에서 T2/T2* 맵
             - MREPT(mems, 스핀에코 10에코) → **T2**  : 1·2 both, PRE·POST 모두
             - post_T2_star_AX-2(mgems, 그래디언트에코 10에코) → **T2\\*** : 개체 2 POST만
  C. EPT   : reconphs 위상 → 라플라시안 → 전도도 sigma = lap(phi) / (2 mu0 omega)

실행: python _run_0820.py            (1, 2 모두)
      python _run_0820.py 1          (하나만)

--------------------------------------------------------------------------
반드시 확인이 필요한 상수 두 개 (아래 CONFIG)
--------------------------------------------------------------------------
1) DTE_MS  — 에코 간격.
   DICOM에 에코별 EchoTime이 안 들어 있다. 100개 파일 전부 EchoTime이 첫 에코 값
   (mems 7.0ms, mgems 3.0ms)으로 동일하게 찍혀 있고, 에코를 구분하는 태그는
   EchoNumber(1~10)뿐이다. 그래서 TE_n = n x DTE_MS 로 가정했다.
   ms 단위 T2/T2*는 전부 이 상수에 **선형 비례**한다. 실제 간격을 알면 그 값만 바꾸면
   되고, 에코 지수 단위 결과(R2_per_echo)는 이 가정과 무관하다.

2) PHASE_UNIT_PER_RAD — reconphs 파일의 위상 단위.
   근거: 같은 변환기가 만든 magnitude nii의 최대값/100 = 3714.86 이 DICOM 스케일 태그로
   계산한 32767/8.8206 = 3714.81 과 5자리까지 일치한다. 즉 nii = 참값 x 100 이다.
   같은 규칙을 위상에 적용하면 phi[rad] = reconphs/100 이고, 이때 위상 범위는 ±11.7 rad
   (약 7.4 pi)로 400MHz 송수신 위상으로 무리 없는 값이다.
   sigma는 이 상수에 **반비례**한다. reconphs를 만든 재구성 코드에서 확인할 것.
   참고로 스크립트는 대안값(370)일 때의 sigma도 함께 출력한다.
"""
import os
import re
import sys
import csv
import glob
from dataclasses import replace
import numpy as np
import pydicom
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ CONFIG
ROOT = r"C:\Users\user\Desktop\대학원\6. 원자력의학원\26-08-20"
DATA = os.path.join(ROOT, "260820_BBB_RAT_EPT")
OUTROOT = os.path.join(ROOT, "2. 분석 결과")
CODE_DIR = r"C:\Users\user\Desktop\대학원\6. 원자력의학원\MRI 분석 코드"
SCALE_TAG = (0x00E1, 0x1001)

DTE_MS = {"mems": 7.0, "mgems": 3.0}   # 에코 간격 가정 (= 각 시퀀스의 첫 TE)
PHASE_UNIT_PER_RAD = 100.0             # reconphs 단위/rad
PHASE_UNIT_ALT = 370.0                 # 대안 가정 (참고 출력용)
MU0 = 4e-7 * np.pi
EPT_RADII = (9, 21, 31)                # 라플라시안 커널 반경 (화소). 크게 잡을수록 저잡음·저해상도
HEAT_VMAX = 100

sys.path.insert(0, CODE_DIR)
from mri_dce import (Slab, Volume, register_slab, enhancement_curve, plot_curves,
                     save_frame_slices, heatmap_frames_row, subtraction_frames_row,
                     montage_anatomy, montage_heatmap, classify_orientation)


# ------------------------------------------------------------------ 공통
def read_series(d):
    """슬라이스 폴더 하나 → (data[z,y,x] 스케일보정, meta dict)."""
    fs = sorted(os.listdir(d))
    per = {}
    ds0 = None
    for f in fs:
        ds = pydicom.dcmread(os.path.join(d, f))
        ds0 = ds0 or ds
        sl = int(re.search(r"slice(\d+)", f).group(1))
        ec = int(re.search(r"echo(\d+)", f).group(1))
        per[(ec, sl)] = ds.pixel_array.astype(np.float32)
    sc = float(ds0[SCALE_TAG].value)
    echoes = sorted({k[0] for k in per})
    slices = sorted({k[1] for k in per})
    arr = np.stack([np.stack([per[(e, s)] for s in slices]) for e in echoes]) / sc
    iop = [float(x) for x in ds0.ImageOrientationPatient]
    ps = [float(x) for x in ds0.PixelSpacing]
    meta = dict(scale=sc, n_echo=len(echoes), n_slice=len(slices), iop=iop,
                spacing=(ps[1], ps[0], float(ds0.SliceThickness)),
                seq=str(getattr(ds0, "SequenceName", "?")),
                tr=float(ds0.RepetitionTime), te1=float(ds0.EchoTime),
                fa=float(ds0.FlipAngle), freq=float(ds0.ImagingFrequency),
                ipp=[float(x) for x in ds0.ImagePositionPatient],
                orientation=classify_orientation(iop))
    return arr, meta   # arr[echo, slice, y, x]


def series_ipp_min(d):
    """시리즈의 최소투영 슬라이스 IPP (기하 정렬용)."""
    best = None
    for f in sorted(os.listdir(d)):
        ds = pydicom.dcmread(os.path.join(d, f), stop_before_pixels=True)
        iop = np.array([float(x) for x in ds.ImageOrientationPatient])
        n = np.cross(iop[:3], iop[3:])
        ipp = np.array([float(x) for x in ds.ImagePositionPatient])
        p = float(np.dot(ipp, n))
        if best is None or p < best[0]:
            best = (p, ipp)
    return best[1]


def tp_rank(lab):
    if lab == "PRE":
        return 0
    if lab == "POST":
        return 1
    m = re.search(r"GD(\d+)", lab)
    return 1 + int(m.group(1))


def gre_series(rat):
    """개체의 GRE 시리즈 목록 → {ori: [(label, path), ...]} 시간순."""
    out = {}
    for sub in ("PRE", "POST"):
        p = os.path.join(DATA, rat, sub)
        for d in sorted(os.listdir(p)):
            if not d.endswith(".dcm") or "GRE" not in d:
                continue
            ori = "AX" if "_AX_" in d else "COR"
            lab = d.replace(f"0820_GRE_{ori}_", "").split("_NO")[0]
            out.setdefault(ori, []).append((lab, os.path.join(p, d)))
    for ori in out:
        out[ori].sort(key=lambda x: tp_rank(x[0]))
    return out


def to_grid(data, meta, ref_ipp, ref_meta):
    """물리좌표 평행이동으로 ref 격자에 리샘플. 0.01 미만 성분은 스냅."""
    r = np.array(ref_meta["iop"][:3]); c = np.array(ref_meta["iop"][3:])
    n = np.cross(r, c)
    d = ref_ipp - np.array(meta["_ippmin"])
    sx, sy, sz = ref_meta["spacing"]
    dk, dj, di = float(np.dot(d, n)) / sz, float(np.dot(d, c)) / sy, float(np.dot(d, r)) / sx
    dk, dj, di = (0.0 if abs(x) < 0.01 else x for x in (dk, dj, di))
    if dk == 0.0 and dj == 0.0 and di == 0.0:
        return data, dk
    return ndimage.shift(data, (-dk, -dj, -di), order=1, mode="constant", cval=0.0), dk


def save_map(arr, out, title, cmap, vmin, vmax, label, mask=None, under=None):
    a = np.where(mask, arr, np.nan) if mask is not None else arr
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    if under is not None:
        ax.imshow(under, cmap="gray", vmin=np.percentile(under, 1), vmax=np.percentile(under, 99))
    im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off"); ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=label)
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)


def save_montage(vol3, out, title, cmap, vmin, vmax, label, mask=None):
    n = vol3.shape[0]
    cols = min(5, n); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.0 * cols, 3.1 * rows), squeeze=False)
    axes = axes.ravel(); im = None
    for z in range(n):
        a = np.where(mask[z], vol3[z], np.nan) if mask is not None else vol3[z]
        im = axes[z].imshow(a, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[z].set_title(f"slice {z}", fontsize=8)
    for a in axes:
        a.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(right=0.9)
    fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.010, 0.7]), label=label)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ A. DCE
def run_dce(rat, outdir):
    print(f"  --- A. DCE ---", flush=True)
    os.makedirs(outdir, exist_ok=True)
    rows, summary, slabs = [], [], []
    for ori, items in sorted(gre_series(rat).items()):
        vols, metas = {}, {}
        for lab, p in items:
            a, m = read_series(p)
            m["_ippmin"] = series_ipp_min(p)
            vols[lab] = a[0]           # GRE는 단일 에코
            metas[lab] = m
        ref = "POST"                    # Gd 기준선 = FUS 후·Gd 전
        refm = metas[ref]; refipp = refm["_ippmin"]
        merged, dks = {}, {}
        for lab in vols:
            merged[lab], dks[lab] = to_grid(vols[lab], metas[lab], refipp, refm)
        labs = sorted(vols, key=tp_rank)

        def mkvol(lab, order):
            m = metas[lab]
            return Volume(data=merged[lab].astype(np.float32), spacing=refm["spacing"],
                          origin=tuple(refm["_ippmin"]), iop=tuple(refm["iop"]),
                          orientation=refm["orientation"], series_uid=lab,
                          acq_time=None, order=order, condition=lab, tp=tp_rank(lab))

        gd = [l for l in labs if l != "PRE"]          # POST, GD1..GDn
        slab = Slab(volumes=[mkvol(l, i) for i, l in enumerate(gd)],
                    orientation=refm["orientation"], animal=f"NO{rat}")
        slab = register_slab(slab, "inplane", True, verbose=True)

        tis = slab.pre.data > float(np.mean(slab.pre.data[slab.pre.data > 0]))
        support = np.all(np.stack([v.data for v in slab.volumes]) > 0, axis=0)
        mask = tis & support
        slabs.append((slab, gd, mask))
        drop = 100.0 * (1 - mask.sum() / max(int(tis.sum()), 1))
        cv = enhancement_curve(slab, mask)
        print(f"    {ori}: 시점 {gd}", flush=True)
        print(f"    {ori}: Gd 증강(기준 POST) = {cv['enh_pct']}  peak={cv['peak_pct']}%  "
              f"(0채움 제외 {drop:.1f}%)", flush=True)
        summary.append([f"NO{rat}", ori, "Gd(기준=POST)", " ".join(gd),
                        " ".join(f"{x:+.1f}" for x in cv["enh_pct"]),
                        cv["peak_pct"], round(drop, 2)])

        # PRE -> POST (FUS 후·Gd 전) 비교
        pre_a, post_a = merged["PRE"], merged["POST"]
        sup2 = (pre_a > 0) & (post_a > 0)
        m2 = tis & sup2
        pp = float(post_a[m2].mean() / pre_a[m2].mean() - 1) * 100
        summary.append([f"NO{rat}", ori, "PRE->POST(Gd 전)", "PRE POST",
                        f"{pp:+.1f}", round(pp, 1),
                        round(100 * (1 - m2.sum() / max(int(tis.sum()), 1)), 2)])
        print(f"    {ori}: PRE→POST(Gd 전) 조직 신호 {pp:+.1f}%  "
              f"(면외 이동 PRE {dks['PRE']:+.2f} slice)", flush=True)
        rows.append([f"NO{rat}", ori, refm["spacing"][2], len(labs),
                     round(dks["PRE"], 3), " ".join(labs)])

        tag = slab.label()
        montage_anatomy(slab, "pre", os.path.join(outdir, f"{tag}_POST_anatomy.png"))
        montage_anatomy(slab, "peak", os.path.join(outdir, f"{tag}_lastGD_anatomy.png"))
        montage_heatmap(slab, os.path.join(outdir, f"{tag}_heatmap.png"))
        d_ind = os.path.join(outdir, ori, "individual")
        d_hm = os.path.join(outdir, ori, "heatmap_rows")
        d_sub = os.path.join(outdir, ori, "subtraction_rows")
        for d in (d_ind, d_hm, d_sub):
            os.makedirs(d, exist_ok=True)
        wl = np.percentile(slab.volumes[-1].data, [1, 99])
        for z in range(slab.n_slices):
            save_frame_slices(slab, z, d_ind, wl=wl)
            heatmap_frames_row(slab, z, os.path.join(d_hm, f"{tag}_slice{z:02d}_heatmap_row.png"))
            subtraction_frames_row(slab, z, os.path.join(d_sub, f"{tag}_slice{z:02d}_subtraction_row.png"))

        # 마지막 GD - POST 차감/히트맵
        d_p = os.path.join(outdir, "post_vs_lastGD")
        os.makedirs(os.path.join(d_p, "heatmap"), exist_ok=True)
        pre = slab.volumes[0].data.astype(float)
        last = slab.volumes[-1].data.astype(float)
        delta = last - pre
        enh = np.where(mask, delta / (pre + 1e-6) * 100.0, np.nan)
        M = float(np.percentile(np.abs(delta[mask]), 99)) or 1.0
        wl2 = np.percentile(last, [1, 99])
        for z in range(slab.n_slices):
            save_map(delta[z], os.path.join(d_p, f"{tag}_slice{z:02d}_delta.png"),
                     f"{tag} slice {z}  dS = {gd[-1]} - POST", "seismic", -M, M, "dS")
            save_map(enh[z], os.path.join(d_p, "heatmap", f"{tag}_slice{z:02d}_heat.png"),
                     f"{tag} slice {z}  ({gd[-1]}-POST)/POST", "jet", 0, HEAT_VMAX,
                     "% 증강", under=last[z])
        save_montage(delta, os.path.join(d_p, f"{tag}_montage_delta.png"),
                     f"{tag}  dS = {gd[-1]} - POST", "seismic", -M, M, "dS")
        save_montage(enh, os.path.join(d_p, f"{tag}_montage_heat.png"),
                     f"{tag}  ({gd[-1]}-POST)/POST %", "jet", 0, HEAT_VMAX, "% 증강")

    with open(os.path.join(outdir, "dce_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rat", "orientation", "비교", "시점", "증강_%_by_frame", "peak_%", "0채움제외_%"])
        w.writerows(summary)
    with open(os.path.join(outdir, "dce_geometry.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rat", "orientation", "slice_mm", "n_timepoints", "PRE_면외이동_slice", "시점"])
        w.writerows(rows)
    plot_curves([enhancement_curve(s, mk) for s, _, mk in slabs],
                os.path.join(outdir, "all_curves.png"))
    return summary


# ------------------------------------------------------------------ B. T2 / T2*
def fit_relax(arr, dte_ms, noise):
    """arr[echo, z, y, x] → (R2_per_echo, T2_ms, S0, 사용에코수). 가중 로그선형."""
    ne = arr.shape[0]
    n = np.arange(1, ne + 1, dtype=float)
    good = arr > (3.0 * noise)
    S = np.where(good, arr, np.nan)
    L = np.log(np.clip(S, 1e-6, None))
    W = np.where(good, arr, 0.0) ** 2          # 로그변환 잡음 보정 가중치
    sw = W.sum(0)
    nx = (W * n[:, None, None, None]).sum(0) / np.where(sw > 0, sw, 1)
    ly = np.nansum(W * L, 0) / np.where(sw > 0, sw, 1)
    dn = n[:, None, None, None] - nx
    dl = L - ly
    num = np.nansum(W * dn * dl, 0)
    den = np.nansum(W * dn * dn, 0)
    slope = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    R2e = -slope                                # 에코당 감쇠율
    with np.errstate(divide="ignore", invalid="ignore"):
        T2 = np.where(R2e > 0, dte_ms / R2e, np.nan)
    S0 = np.exp(ly + R2e * nx)
    return R2e, T2, S0, good.sum(0)


def run_relax(rat, outdir):
    print(f"  --- B. T2 / T2* ---", flush=True)
    os.makedirs(outdir, exist_ok=True)
    rows = []
    jobs = []
    for sub in ("PRE", "POST"):
        p = os.path.join(DATA, rat, sub)
        for d in sorted(os.listdir(p)):
            if not d.endswith(".dcm"):
                continue
            if "MREPT" in d:
                jobs.append((f"MREPT_{sub}", "T2", os.path.join(p, d)))
            elif "T2_star" in d:
                jobs.append((f"T2star_{sub}", "T2*", os.path.join(p, d)))
    for name, kind, p in jobs:
        arr, m = read_series(p)
        if m["n_echo"] < 3:
            print(f"    [건너뜀] {name}: 에코 {m['n_echo']}개", flush=True)
            continue
        dte = DTE_MS.get(m["seq"], m["te1"])
        noise = float(np.median(arr[-1][arr[-1] < np.percentile(arr[-1], 20)])) * 1.4826
        R2e, T2, S0, nused = fit_relax(arr, dte, max(noise, 1e-6))
        mag1 = arr[0]
        tis = mag1 > float(np.mean(mag1[mag1 > 0]))
        ok = tis & np.isfinite(T2) & (nused >= 4) & (T2 > 0) & (T2 < 10 * dte * m["n_echo"])
        vals = T2[ok]
        print(f"    {name} ({m['seq']}, {kind}): 에코 {m['n_echo']}개, TR={m['tr']} FA={m['fa']} "
              f"TE1={m['te1']}ms, dTE 가정 {dte}ms", flush=True)
        print(f"      조직 {kind} 중앙값 = {np.median(vals):.2f} ms "
              f"(IQR {np.percentile(vals,25):.2f}~{np.percentile(vals,75):.2f}), "
              f"유효화소 {ok.sum()}", flush=True)
        rows.append([f"NO{rat}", name, m["seq"], kind, m["n_echo"], m["tr"], m["fa"],
                     m["te1"], dte, int(ok.sum()),
                     round(float(np.median(vals)), 3),
                     round(float(np.percentile(vals, 25)), 3),
                     round(float(np.percentile(vals, 75)), 3),
                     round(float(np.median(R2e[ok])), 4)])
        vmax = float(np.percentile(vals, 95))
        save_montage(np.where(ok, T2, np.nan), os.path.join(outdir, f"{name}_{kind.replace('*','star')}_map.png"),
                     f"NO{rat} {name}  {kind} map (dTE={dte}ms 가정)", "viridis", 0, vmax, f"{kind} (ms)")
        save_montage(np.where(ok, R2e, np.nan), os.path.join(outdir, f"{name}_R2_per_echo.png"),
                     f"NO{rat} {name}  에코당 감쇠율 R2 (가정 무관)", "magma", 0,
                     float(np.percentile(R2e[ok], 95)), "1/echo")
        np.savez_compressed(os.path.join(outdir, f"{name}_maps.npz"),
                            T2_ms=T2.astype(np.float32), R2_per_echo=R2e.astype(np.float32),
                            S0=S0.astype(np.float32), mask=ok, mag_echo1=mag1.astype(np.float32),
                            dte_ms=dte, n_echo=m["n_echo"], spacing=np.array(m["spacing"]))
        # 에코 감쇠 곡선
        fig, ax = plt.subplots(figsize=(6, 4))
        prof = [float(np.median(arr[e][ok])) for e in range(m["n_echo"])]
        ax.semilogy(np.arange(1, m["n_echo"] + 1), prof, "o-")
        ax.set_xlabel("에코 번호"); ax.set_ylabel("조직 중앙 신호 (log)")
        ax.set_title(f"NO{rat} {name} 에코 감쇠"); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, f"{name}_decay.png"), dpi=130)
        plt.close(fig)
    with open(os.path.join(outdir, "relax_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rat", "series", "sequence", "kind", "n_echo", "TR_ms", "FA_deg",
                    "TE1_ms_from_dicom", "dTE_ms_assumed", "n_vox",
                    "median_ms", "q25_ms", "q75_ms", "median_R2_per_echo"])
        w.writerows(rows)
    return rows


# ------------------------------------------------------------------ C. EPT
def lap_kernel(R, hx, hy):
    """반경 R 원판 위 2차식 최소제곱 적합의 라플라시안 추출 커널 [1/m^2].
    화소별 미분보다 훨씬 잡음에 강하다(표준 kernel-based EPT)."""
    ys, xs = np.mgrid[-R:R + 1, -R:R + 1]
    m = (xs ** 2 + ys ** 2) <= R * R
    X = np.stack([np.ones(m.sum()), xs[m], ys[m], xs[m] ** 2, ys[m] ** 2, xs[m] * ys[m]],
                 1).astype(float)
    C = np.linalg.pinv(X)
    w = np.zeros((2 * R + 1, 2 * R + 1))
    w[m] = 2 * C[3] / hx ** 2 + 2 * C[4] / hy ** 2
    return w


def run_ept(rat, outdir):
    print(f"  --- C. EPT ---", flush=True)
    os.makedirs(outdir, exist_ok=True)
    res, maps = [], {}
    for sub in ("PRE", "POST"):
        p = os.path.join(DATA, rat, sub)
        ph = glob.glob(os.path.join(p, "*reconphs.nii"))
        mg = glob.glob(os.path.join(p, "*MREPT*.nii"))
        if not ph or not mg:
            print(f"    [건너뜀] {sub}: reconphs 또는 magnitude 없음", flush=True)
            continue
        dcm = glob.glob(os.path.join(p, "*MREPT*.dcm"))[0]
        _, m = read_series(dcm)
        freq = m["freq"] * 1e6
        omega = 2 * np.pi * freq
        sx, sy, sz = [v * 1e-3 for v in m["spacing"]]      # mm → m

        def stack(d):
            fs = sorted(f for f in os.listdir(d) if f.startswith("vol_") and f.endswith(".nii"))
            return np.stack([np.asarray(nib.load(os.path.join(d, f)).dataobj, dtype=np.float64)
                             for f in fs])                  # [echo, y, x, z]

        Pe, Ae = stack(ph[0]), stack(mg[0])
        # 에코 평균은 magnitude 가중. SE라 송수신 위상은 에코 무관해야 하고,
        # 실측 위상잡음이 에코1 2.04 rad → 가중평균 0.80 rad로 줄어든다.
        W = Ae / np.clip(Ae.sum(0, keepdims=True), 1e-9, None)
        P = (Pe * W).sum(0)
        A = Ae.mean(0)
        P = np.moveaxis(P, 2, 0); A = np.moveaxis(A, 2, 0)   # → [z, y, x]
        phi = P / PHASE_UNIT_PER_RAD
        tis = A > float(np.mean(A[A > 0]))
        core = ndimage.binary_erosion(tis, np.ones((1, 7, 7)))

        # 위상잡음 실측 (고주파 성분)
        nse = float((phi - ndimage.gaussian_filter(phi, (0, 2, 2)))[tis].std())
        K = 2 * MU0 * omega
        print(f"    {sub}: f={freq/1e6:.2f}MHz, 화소 {m['spacing'][0]:.4f}mm, "
              f"2*mu0*omega={K:.1f}", flush=True)
        print(f"      위상 범위 {phi.min():.2f}~{phi.max():.2f} rad "
              f"(단위가정 {PHASE_UNIT_PER_RAD}/rad), 실측 위상잡음 {nse:.3f} rad", flush=True)

        sig_maps = {}
        for R in EPT_RADII:
            w = lap_kernel(R, sx, sy)
            lap = np.stack([ndimage.convolve(phi[z], w, mode="nearest")
                            for z in range(phi.shape[0])])
            sg = lap / K
            sig_maps[R] = (sg, lap)
            pred = nse * float(np.linalg.norm(w)) / K       # 이론 잡음
            v = sg[core]
            print(f"      R={R:2d}vox({R*m['spacing'][0]:.2f}mm): 중앙값 {np.median(v):+.3f} S/m, "
                  f"IQR {np.percentile(v,25):+.3f}~{np.percentile(v,75):+.3f}, "
                  f"이론잡음 ±{pred:.3f} S/m", flush=True)
            res.append([f"NO{rat}", sub, round(freq / 1e6, 3), PHASE_UNIT_PER_RAD, R,
                        round(R * m["spacing"][0], 3), round(nse, 4), int(core.sum()),
                        round(float(np.median(v)), 4),
                        round(float(np.percentile(v, 25)), 4),
                        round(float(np.percentile(v, 75)), 4),
                        round(pred, 4),
                        round(float(np.median(v)) * PHASE_UNIT_PER_RAD / PHASE_UNIT_ALT, 4)])
            lim = max(float(np.percentile(np.abs(v), 90)), 1e-6)
            save_montage(sg, os.path.join(outdir, f"NO{rat}_{sub}_sigma_R{R:02d}.png"),
                         f"NO{rat} {sub}  EPT 전도도  R={R}vox ({R*m['spacing'][0]:.2f}mm), "
                         f"이론잡음 ±{pred:.2f} S/m", "seismic", -lim, lim,
                         "sigma (S/m)", mask=core)
        maps[sub] = (sig_maps, tis, core, A, phi)

        save_montage(phi, os.path.join(outdir, f"NO{rat}_{sub}_phase.png"),
                     f"NO{rat} {sub}  송수신 위상 (에코 magnitude 가중평균)", "twilight",
                     float(np.percentile(phi, 1)), float(np.percentile(phi, 99)), "rad")
        np.savez_compressed(
            os.path.join(outdir, f"NO{rat}_{sub}_ept.npz"),
            phase_rad=phi.astype(np.float32), phase_raw=P.astype(np.float32),
            magnitude=A.astype(np.float32), tissue_mask=tis, core_mask=core,
            phase_noise_rad=nse, phase_unit_per_rad=PHASE_UNIT_PER_RAD,
            freq_Hz=freq, spacing_mm=np.array(m["spacing"]),
            **{f"sigma_R{R:02d}_S_per_m": sig_maps[R][0].astype(np.float32) for R in EPT_RADII},
            **{f"laplacian_R{R:02d}_rad_per_m2": sig_maps[R][1].astype(np.float32) for R in EPT_RADII})

    if "PRE" in maps and "POST" in maps:
        smp, tp, cp, _, _ = maps["PRE"]
        smo, to, co, _, _ = maps["POST"]
        mm = cp & co
        for R in EPT_RADII:
            if smp[R][0].shape != smo[R][0].shape:
                print("    [경고] PRE/POST shape 불일치 → 차이맵 생략", flush=True)
                break
            d = smo[R][0] - smp[R][0]
            lim = max(float(np.percentile(np.abs(d[mm]), 90)), 1e-6)
            save_montage(d, os.path.join(outdir, f"NO{rat}_dsigma_POST_minus_PRE_R{R:02d}.png"),
                         f"NO{rat}  d(sigma) = POST - PRE, R={R}vox", "seismic",
                         -lim, lim, "d sigma (S/m)", mask=mm)
            print(f"    R={R:2d}: d(sigma) POST-PRE 중앙값 = {np.median(d[mm]):+.4f} S/m "
                  f"(IQR {np.percentile(d[mm],25):+.3f}~{np.percentile(d[mm],75):+.3f})", flush=True)
            res.append([f"NO{rat}", "POST-PRE", "", PHASE_UNIT_PER_RAD, R, "", "",
                        int(mm.sum()), round(float(np.median(d[mm])), 4),
                        round(float(np.percentile(d[mm], 25)), 4),
                        round(float(np.percentile(d[mm], 75)), 4), "", ""])

    with open(os.path.join(outdir, "ept_summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rat", "timepoint", "freq_MHz", "phase_unit_per_rad", "kernel_R_vox",
                    "kernel_R_mm", "phase_noise_rad", "core_vox", "sigma_median_S_per_m",
                    "sigma_q25", "sigma_q75", "예측잡음_S_per_m", "sigma_median_대안가정370"])
        w.writerows(res)
    return res


# ------------------------------------------------------------------ 실행
rats = sys.argv[1:] or ["1", "2"]
for rat in rats:
    print(f"\n########## RAT {rat} ##########", flush=True)
    out = os.path.join(OUTROOT, rat)
    run_dce(rat, os.path.join(out, "dce"))
    run_relax(rat, os.path.join(out, "relax"))
    run_ept(rat, os.path.join(out, "ept"))
print("\n완료", flush=True)
