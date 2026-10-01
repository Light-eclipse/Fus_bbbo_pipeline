# -*- coding: utf-8 -*-
"""261001 폴더 1 reconphs — 8에코 평균에 보이는 구조가 실재하는가.

1차 진단(`_ept_diag_1001.py`)에서 나온 것
  * 낱 에코 위상은 조직에서도 잡음 SD 2.03 rad. magnitude SNR 157 이면
    0.0064 rad 여야 하므로 318배 크다.
  * 값이 +-12.3 으로 +-pi 를 넘는다.
  * 그런데 8에코를 평균하면 해부 구조처럼 보이는 것이 나타난다.

마지막 항목이 애매하다. 잡음을 평균해도 표시 범위가 좁아지면 구조처럼 보일 수
있다. 그래서 **독립 반쪽 시험**을 한다. 홀수 에코 {1,3,5,7} 평균과 짝수 에코
{2,4,6,8} 평균은 서로 독립인 두 측정이다.

  * 구조가 실재하면 두 반쪽이 상관을 보인다.
  * 잡음이면 상관이 0 이다.

같은 시험을 magnitude 에도 돌려 대조군으로 쓴다(magnitude 는 당연히 상관이
높아야 하고, 그렇지 않으면 시험 자체가 잘못된 것이다).

덧붙여 세 가지를 더 본다.
  * 에코1-에코2 voxel 상관 (재현성)
  * 위상 잡음이 1/SNR 을 따르는가 (진짜 위상이면 따라야 한다)
  * 조직 대 공기 잡음비 (마스크를 제대로 잡아서 다시)

실행: python "_ept_split_1001.py"
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


def read_nii(f):
    with open(f, "rb") as fh:
        h = nib.Nifti1Header.from_fileobj(fh)
        a = np.asarray(h.data_from_fileobj(fh), np.float64)
    sl = float(h["scl_slope"]) if h["scl_slope"] else 1.0
    return a / sl


def load(d):
    return np.stack([read_nii(f) for f in
                     sorted(glob.glob(os.path.join(d, "vol_image001echo*.nii")))])


def corr(a, b, m):
    x, y = a[m], b[m]
    return float(np.corrcoef(x, y)[0, 1])


mag = load(os.path.join(DATA, "1", "1001_MREPT-NO1_20261001_01.nii"))
phs = load(os.path.join(DATA, "1", "reconphs.nii"))

# ---- 마스크: 조직은 magnitude 상위, 공기는 조직에서 충분히 떨어진 저신호
m1 = mag[0]
tissue = ndimage.binary_erosion(m1 > np.percentile(m1, 55), np.ones((3, 3, 1)),
                                iterations=2)
far = ~ndimage.binary_dilation(m1 > np.percentile(m1, 35), np.ones((5, 5, 1)),
                               iterations=3)
air = far & (m1 < np.percentile(m1, 20))
print(f"조직 {int(tissue.sum())}화소   공기 {int(air.sum())}화소")
print(f"magnitude 조직 중앙 {np.median(m1[tissue]):.1f}   "
      f"공기 중앙 {np.median(m1[air]):.2f}  SD {np.std(m1[air]):.2f}")

# ------------------------------------------------- 1. 조직 대 공기 위상 잡음
print("\n" + "=" * 76)
print("1. 위상 잡음이 신호 세기를 따라가는가")
print("   진짜 위상이면 잡음 = 1/SNR 이다. 조직은 작고 공기는 pi/sqrt(3)=1.814 다.")
print(f"   {'에코':<5}{'SNR':>8}{'기대 잡음':>11}{'조직 실측':>11}{'공기 실측':>11}{'실측/기대':>11}")


def nsd(v, m):
    d = np.diff(v, axis=0)
    mm = m[1:] & m[:-1]
    return float(np.std(d[mm]) / np.sqrt(2)) if mm.sum() > 50 else float("nan")


rows = []
for e in range(8):
    sig = float(np.median(mag[e][tissue]))
    nz = float(np.std(mag[e][air]))
    snr = sig / nz * np.sqrt(2 - np.pi / 2)
    exp = 1.0 / snr
    nt, na = nsd(phs[e], tissue), nsd(phs[e], air)
    rows.append((e + 1, snr, exp, nt, na))
    print(f"   {e+1:<5}{snr:>8.1f}{exp:>11.4f}{nt:>11.4f}{na:>11.4f}{nt/exp:>11.0f}")

# ----------------------------------------------- 2. 에코 간 재현성 (voxel 상관)
print("\n" + "=" * 76)
print("2. 에코 사이에 같은 값이 재현되는가 (조직 내 voxel 상관)")
print("   스핀에코 송수신 위상은 에코에 무관하므로 상관이 1 에 가까워야 한다.")
for e in range(1, 8):
    print(f"   위상 에코1 vs 에코{e+1}: r = {corr(phs[0], phs[e], tissue):+.4f}"
          f"      (대조: magnitude r = {corr(mag[0], mag[e], tissue):+.4f})")

# ------------------------------------------- 3. 독립 반쪽 시험 (핵심)
print("\n" + "=" * 76)
print("3. 독립 반쪽 시험 — 8에코 평균의 구조가 실재하는가")
po, pe = phs[0::2].mean(0), phs[1::2].mean(0)
mo, me = mag[0::2].mean(0), mag[1::2].mean(0)
rp, rm = corr(po, pe, tissue), corr(mo, me, tissue)
print(f"   위상     홀수에코 평균 vs 짝수에코 평균 : r = {rp:+.4f}")
print(f"   magnitude 홀수에코 평균 vs 짝수에코 평균 : r = {rm:+.4f}   (대조군)")
print("   -> " + ("위상에도 재현되는 공간 구조가 있다" if rp > 0.3 else
                  "위상 반쪽끼리 상관이 없다. 평균 영상의 구조는 실재가 아니다"))

# 공간 저역통과 후에도 남는가 (국소 잡음을 지우고 큰 구조만 본다)
sm = lambda v: ndimage.gaussian_filter(v, (3, 3, 0))
rp_s, rm_s = corr(sm(po), sm(pe), tissue), corr(sm(mo), sm(me), tissue)
print(f"   3화소 평활 후:  위상 r = {rp_s:+.4f}   magnitude r = {rm_s:+.4f}")

# ------------------------------------------------- 4. magnitude 와의 관계
print("\n" + "=" * 76)
print("4. 위상이 magnitude 와 얽혀 있는가")
print(f"   위상 8에코 평균 vs magnitude 에코1 : r = {corr(phs.mean(0), mag[0], tissue):+.4f}")
print(f"   |위상| 8에코 평균 vs magnitude 에코1 : r = "
      f"{corr(np.abs(phs).mean(0), mag[0], tissue):+.4f}")

# -------------------------------------------------------------------- 그림
zc = mag.shape[3] // 2
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
panels = [("magnitude 에코1", m1[:, :, zc], "gray", None),
          ("위상 홀수에코 평균", po[:, :, zc], "twilight", 4),
          ("위상 짝수에코 평균", pe[:, :, zc], "twilight", 4),
          ("위상 8에코 평균", phs.mean(0)[:, :, zc], "twilight", 4)]
for j, (t, v, cm, lim) in enumerate(panels):
    kw = dict(vmin=-lim, vmax=lim) if lim else {}
    im = ax[0, j].imshow(v.T, cmap=cm, **kw)
    ax[0, j].set_title(t, fontsize=10)
    ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
    fig.colorbar(im, ax=ax[0, j], fraction=0.046)

ax[1, 0].plot([r[0] for r in rows], [r[2] for r in rows], "o-", label="1/SNR (기대)")
ax[1, 0].plot([r[0] for r in rows], [r[3] for r in rows], "s-", label="조직 실측")
ax[1, 0].plot([r[0] for r in rows], [r[4] for r in rows], "^-", label="공기 실측")
ax[1, 0].axhline(np.pi / np.sqrt(3), color="k", ls="--", lw=0.8, label="균일난수 1.814")
ax[1, 0].set_yscale("log"); ax[1, 0].set_xlabel("에코")
ax[1, 0].set_ylabel("위상 잡음 SD (rad)"); ax[1, 0].legend(fontsize=8)
ax[1, 0].grid(alpha=0.3)

idx = np.random.default_rng(0).choice(int(tissue.sum()), 4000, replace=False)
ax[1, 1].plot(po[tissue][idx], pe[tissue][idx], ".", ms=1.5, alpha=0.4)
ax[1, 1].set_xlabel("홀수에코 평균"); ax[1, 1].set_ylabel("짝수에코 평균")
ax[1, 1].set_title(f"위상 반쪽 시험  r={rp:+.3f}", fontsize=10)
ax[1, 1].grid(alpha=0.3)
ax[1, 2].plot(mo[tissue][idx], me[tissue][idx], ".", ms=1.5, alpha=0.4, color="C1")
ax[1, 2].set_xlabel("홀수에코 평균"); ax[1, 2].set_ylabel("짝수에코 평균")
ax[1, 2].set_title(f"magnitude 반쪽 시험 (대조)  r={rm:+.3f}", fontsize=10)
ax[1, 2].grid(alpha=0.3)
ax[1, 3].hist(phs[0][tissue], bins=100, alpha=0.6, label="조직", density=True)
if air.sum() > 100:
    ax[1, 3].hist(phs[0][air], bins=100, alpha=0.6, label="공기", density=True)
ax[1, 3].set_title("에코1 위상 분포", fontsize=10)
ax[1, 3].legend(fontsize=8); ax[1, 3].grid(alpha=0.3)

fig.suptitle("261001 폴더 1 — reconphs 독립 반쪽 시험", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "reconphs_split.png"), dpi=130, bbox_inches="tight")
print("\n저장:", os.path.join(OUT, "reconphs_split.png"))
