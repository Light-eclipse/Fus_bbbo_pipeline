# -*- coding: utf-8 -*-
"""261001 폴더 1 reconphs — 매끄러운 성분과 잡음을 분리하고 정체를 가린다.

반쪽 시험에서 나온 것
  * 낱 voxel 위상 잡음 2.06 rad (magnitude SNR 로는 0.006 rad 여야 한다)
  * 그런데 3화소 평활 후 홀수에코 평균 대 짝수에코 평균 r = 0.9965
  -> 위상 = **실재하는 매끄러운 장** + **아주 큰 화소 잡음** 이다.

그러면 두 가지를 가려야 한다.

A. 매끄러운 장의 크기가 얼마이고 잡음을 평활로 얼마나 눌러야 보이는가
B. 그 장이 송수신 위상인가, B0(자화율) 위상인가

B 를 가르는 방법: 에코에 대한 거동이다.
  * 송수신 위상은 TE 에 무관하다 (상수항)
  * B0 위상은 스핀에코면 재수렴해 상수, 그래디언트에코면 TE 에 비례해 쌓인다
voxel 마다 위상을 에코 번호로 1차 적합해 **기울기(=B0 성분)와 절편(=TE 0 외삽)**
을 나눈다. 절편이 EPT 가 필요로 하는 송수신 위상의 후보다.

에코1 대 에코8 상관이 0.65 에서 0.49 로 떨어지는 것이 이미 TE 의존 성분이 있다는
신호다. 상수항만 있으면 떨어질 이유가 없다.

실행: python "_ept_decomp_1001.py"
"""
import os
import sys
import glob
import numpy as np
import nibabel as nib
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(BASE))
sys.path.insert(0, os.path.join(ROOT, "MRI 분석 코드"))
from mri_dce import _set_korean_font
_set_korean_font()
plt.rcParams["axes.unicode_minus"] = False

DATA = os.path.join(ROOT, "26-10-01", "261001_BBBO")
OUT = os.path.join(BASE, "ept")
os.makedirs(OUT, exist_ok=True)

PX_MM = 0.273                       # T2_MAP/MREPT 격자 (DICOM 확인값)
SLICE_MM = None                     # 두께는 DICOM 에서 읽는다
GAMMA_HZ = 399.72e6
TWO_MU0_W = 2 * (4e-7 * np.pi) * 2 * np.pi * GAMMA_HZ       # 6312.2


def read_nii(f):
    with open(f, "rb") as fh:
        h = nib.Nifti1Header.from_fileobj(fh)
        a = np.asarray(h.data_from_fileobj(fh), np.float64)
    sl = float(h["scl_slope"]) if h["scl_slope"] else 1.0
    return a / sl


def load(d):
    return np.stack([read_nii(f) for f in
                     sorted(glob.glob(os.path.join(d, "vol_image001echo*.nii")))])


mag = load(os.path.join(DATA, "1", "1001_MREPT-NO1_20261001_01.nii"))
phs = load(os.path.join(DATA, "1", "reconphs.nii"))
nz = mag.shape[3]

# ---------------------------------------------------------------- 마스크
m1 = mag[0]
tissue = ndimage.binary_erosion(m1 > np.percentile(m1, 55), np.ones((3, 3, 1)),
                                iterations=2)
# 공기: FOV 네 귀퉁이 (조직과 절대 겹치지 않는 영역)
air = np.zeros_like(m1, bool)
k = 18
for a0, a1 in ((slice(0, k), slice(0, k)), (slice(0, k), slice(-k, None)),
               (slice(-k, None), slice(0, k)), (slice(-k, None), slice(-k, None))):
    air[a0, a1, :] = True
air &= m1 < np.percentile(m1, 50)
print(f"조직 {int(tissue.sum())}  공기 {int(air.sum())}")
print(f"magnitude  조직 중앙 {np.median(m1[tissue]):8.1f}   "
      f"공기 평균 {np.mean(m1[air]):6.2f}  SD {np.std(m1[air]):6.2f}")
snr1 = np.median(m1[tissue]) / np.std(m1[air]) * np.sqrt(2 - np.pi / 2)
print(f"에코1 조직 SNR {snr1:.1f}  ->  진짜 위상이면 화소 잡음 {1/snr1:.4f} rad")


def nsd(v, m):
    d = np.diff(v, axis=0)
    mm = m[1:] & m[:-1]
    return float(np.std(d[mm]) / np.sqrt(2)) if mm.sum() > 50 else float("nan")


print(f"실측 에코1 위상 잡음  조직 {nsd(phs[0], tissue):.4f}   "
      f"공기 {nsd(phs[0], air):.4f} rad")
print("   -> 조직과 공기가 비슷하면 신호와 무관한 잡음이다")

# ------------------------------------------- A. 매끄러운 장 대 잡음
print("\n" + "=" * 76)
print("A. 평활 반경별 — 매끄러운 장이 얼마나 남고 잡음이 얼마나 줄어드는가")
print(f"   {'sigma(화소)':>11}{'mm':>7}{'장 SD':>10}{'잔차 SD':>10}{'반쪽 r':>9}")
po_raw, pe_raw = phs[0::2].mean(0), phs[1::2].mean(0)
for s in (0, 1, 2, 3, 5, 8, 12):
    f = (lambda v: v) if s == 0 else (lambda v: ndimage.gaussian_filter(v, (s, s, 0)))
    sm = f(phs.mean(0))
    res = phs.mean(0) - sm
    r = float(np.corrcoef(f(po_raw)[tissue], f(pe_raw)[tissue])[0, 1])
    print(f"   {s:>11}{s*PX_MM:>7.2f}{np.std(sm[tissue]):>10.4f}"
          f"{np.std(res[tissue]):>10.4f}{r:>9.4f}")

# ----------------------------------- B. 에코 의존성: 기울기와 절편 분리
print("\n" + "=" * 76)
print("B. 에코에 대한 1차 적합 — 기울기(B0 성분) 대 절편(송수신 위상 후보)")
e = np.arange(1, 9, dtype=float)
X = np.stack([np.ones_like(e), e], 1)
P = np.linalg.pinv(X) @ phs.reshape(8, -1)          # (2, voxel)
icpt = P[0].reshape(phs.shape[1:])
slope = P[1].reshape(phs.shape[1:])
sm = lambda v, s=5: ndimage.gaussian_filter(v, (s, s, 0))
print(f"   절편  조직 SD {np.std(icpt[tissue]):.4f}  평활 후 {np.std(sm(icpt)[tissue]):.4f} rad")
print(f"   기울기 조직 SD {np.std(slope[tissue]):.4f}  평활 후 {np.std(sm(slope)[tissue]):.4f} rad/echo")
print(f"   기울기 조직 중앙 {np.median(slope[tissue]):+.4f} rad/echo")

# 기울기가 실재하는가 — 반쪽으로 확인 (에코 1,3,5,7 과 2,4,6,8 각각 적합)
for nm, idx in (("홀수 에코", [0, 2, 4, 6]), ("짝수 에코", [1, 3, 5, 7])):
    Xi = np.stack([np.ones(4), e[idx]], 1)
    Pi = np.linalg.pinv(Xi) @ phs[idx].reshape(4, -1)
    globals()["ic_" + nm[:2]] = Pi[0].reshape(phs.shape[1:])
    globals()["sl_" + nm[:2]] = Pi[1].reshape(phs.shape[1:])
r_ic = float(np.corrcoef(sm(ic_홀수)[tissue], sm(ic_짝수)[tissue])[0, 1])
r_sl = float(np.corrcoef(sm(sl_홀수)[tissue], sm(sl_짝수)[tissue])[0, 1])
print(f"   반쪽 재현성 (5화소 평활):  절편 r = {r_ic:+.4f}   기울기 r = {r_sl:+.4f}")

# 기울기를 Hz 로 — ΔTE 를 모르므로 가정값 둘을 같이 적는다
for dte in (5.0, 10.0):
    hz = np.std(sm(slope)[tissue]) / (2 * np.pi * dte * 1e-3)
    print(f"   ΔTE {dte:>4.1f} ms 가정 시 기울기 SD = {hz:6.2f} Hz "
          f"({hz/GAMMA_HZ*1e6*1e3:.3f} ppb 수준)")

# ------------------------------------------------ C. 매끄러운 장의 모양
print("\n" + "=" * 76)
print("C. 매끄러운 장이 송수신 위상처럼 생겼는가")
sm_ic = sm(icpt, 5)
print(f"   절편(평활) 조직 범위 {np.percentile(sm_ic[tissue],2):+.3f} ~ "
      f"{np.percentile(sm_ic[tissue],98):+.3f} rad")
print(f"   magnitude 와의 상관 r = "
      f"{np.corrcoef(sm_ic[tissue], m1[tissue])[0,1]:+.4f}  "
      "(송수신 위상은 조직 대비와 무관해야 하므로 0 에 가까워야 한다)")
lap = sum(ndimage.convolve1d(sm_ic, [1., -2., 1.], axis=a) / (PX_MM * 1e-3) ** 2
          for a in (0, 1))
sig = lap / TWO_MU0_W
q = np.percentile(sig[tissue], [25, 50, 75])
print(f"   5화소 평활 라플라시안으로 구한 sigma 분위수 25/50/75 = "
      f"{q[0]:+.2f} / {q[1]:+.2f} / {q[2]:+.2f} S/m")
print(f"   음수 비율 {100*np.mean(sig[tissue] < 0):.1f}%   "
      "(뇌 조직 문헌값은 400MHz 에서 대략 0.5~1 S/m, 음수는 물리적으로 불가)")

# -------------------------------------------------------------------- 그림
zc = nz // 2
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for j, (t, v, cm, lim) in enumerate([
        ("magnitude 에코1", m1[:, :, zc], "gray", None),
        ("위상 8에코 평균", phs.mean(0)[:, :, zc], "twilight", 4),
        ("절편 (TE 0 외삽, 5화소 평활)", sm_ic[:, :, zc], "twilight", 3),
        ("기울기 (rad/echo, 5화소 평활)", sm(slope, 5)[:, :, zc], "coolwarm", 0.3)]):
    kw = dict(vmin=-lim, vmax=lim) if lim else {}
    im = ax[0, j].imshow(v.T, cmap=cm, **kw)
    ax[0, j].set_title(t, fontsize=10)
    ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
    fig.colorbar(im, ax=ax[0, j], fraction=0.046)

ss = [0, 1, 2, 3, 5, 8, 12]
rs = []
for s in ss:
    f = (lambda v: v) if s == 0 else (lambda v, s=s: ndimage.gaussian_filter(v, (s, s, 0)))
    rs.append(float(np.corrcoef(f(po_raw)[tissue], f(pe_raw)[tissue])[0, 1]))
ax[1, 0].plot([s * PX_MM for s in ss], rs, "o-")
ax[1, 0].set_xlabel("평활 sigma (mm)"); ax[1, 0].set_ylabel("반쪽 상관 r")
ax[1, 0].set_title("매끄러운 성분은 재현된다", fontsize=10); ax[1, 0].grid(alpha=0.3)

ax[1, 1].plot(e, [np.median(phs[i][tissue]) for i in range(8)], "o-")
ax[1, 1].set_xlabel("에코"); ax[1, 1].set_ylabel("조직 위상 중앙값 (rad)")
ax[1, 1].set_title("에코에 따른 전체 이동", fontsize=10); ax[1, 1].grid(alpha=0.3)

ax[1, 2].imshow(np.where(tissue[:, :, zc], sig[:, :, zc], np.nan).T,
                cmap="RdBu_r", vmin=-5, vmax=5)
ax[1, 2].set_title("절편으로 구한 sigma (S/m)", fontsize=10)
ax[1, 2].set_xticks([]); ax[1, 2].set_yticks([])

ax[1, 3].hist(sig[tissue], bins=200, range=(-20, 20), color="C3")
ax[1, 3].axvline(0.7, color="k", ls="--", label="문헌 0.7 S/m")
ax[1, 3].set_xlabel("sigma (S/m)"); ax[1, 3].legend(fontsize=8)
ax[1, 3].set_title("sigma 분포", fontsize=10); ax[1, 3].grid(alpha=0.3)

fig.suptitle("261001 폴더 1 — reconphs 성분 분리", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "reconphs_decomp.png"), dpi=130, bbox_inches="tight")
print("\n저장:", os.path.join(OUT, "reconphs_decomp.png"))
