# -*- coding: utf-8 -*-
"""261001 — `reconphs.nii` 가 정말 송수신 위상인지 먼저 판정한다.

왜 이 판정부터 하는가
--------------------
26-08-20 에도 `reconphs.nii` 가 왔고, 거기에 Helmholtz EPT 를 돌리면 조직 안
|lap(phi)| 중앙값이 2.95e5 rad/m^2 로 sigma=0.7 S/m 가 요구하는 4.4e3 의 67배가
나왔다. 조직 절반에서 sigma 가 음수였다. 처음에는 물리 문제로 봤지만, 나중에
**화소 잡음이 조직 1.44 공기 1.58 로 사실상 같다**는 것을 확인했다. 조직 SNR 30
이면 위상 잡음은 0.03 rad 여야 한다. 1.44 rad 는 그보다 50배 크다. 그 파일은
송수신 위상이 아니었다(결합 전 단일 채널인지, 보정 전 위상인지는 미확인).

그래서 숫자를 뽑기 전에 아래 네 가지를 먼저 본다. 하나라도 걸리면 sigma 를
계산하지 않는다. 틀린 전제로 나온 S/m 값은 없는 것만 못하다.

  A. 값의 범위와 분포       — 라디안 위상이면 대략 +-pi, 아니면 다른 단위다
  B. 조직 잡음 vs 공기 잡음 — 결정적 시험. 같으면 위상이 아니다
  C. 에코 간 거동           — 스핀에코라 송수신 위상은 에코에 무관해야 한다.
                              에코에 선형이면 B0/eddy 성분이다
  D. 공간 구조              — 조직 경계를 따라가는가, 아니면 영상과 무관한
                              줄무늬인가

C 가 통과하면 에코 8개 평균으로 위상 잡음이 sqrt(8)=2.83배 줄어든다. 이것이
이번 데이터에서 달라질 수 있는 유일한 지점이라 따로 계산한다.

실행: python "_ept_diag_1001.py"
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

GAMMA_B0_HZ = 399.72e6          # 9.4 T 양성자 공명 주파수 (이전 날짜에서 확인)
MU0 = 4e-7 * np.pi
TWO_MU0_W = 2 * MU0 * 2 * np.pi * GAMMA_B0_HZ      # = 6312 (sigma = lap(phi)/이 값)


def read_nii(f):
    """Agilent NIfTI 는 qform 쿼터니언이 비정규라 nib.load 가 죽는다.
    헤더와 데이터를 직접 읽는다.

    `data_from_fileobj` 는 scl_slope(=100) 를 **곱해서** 돌려준다. 그런데 이
    내보내기 규칙은 `nii = 참값 x 100` 이라 곱이 아니라 나눠야 참값이 된다
    (26-08-20 에서 magnitude nii 최대/100 = 3714.86 이 DICOM 스케일 태그로 구한
    32767/8.8206 = 3714.81 과 5자리까지 맞는 것으로 확인). 그래서 저장된 원값을
    그대로 쓰려고 slope 를 되돌린다."""
    with open(f, "rb") as fh:
        h = nib.Nifti1Header.from_fileobj(fh)
        a = np.asarray(h.data_from_fileobj(fh), np.float64)
    sl = float(h["scl_slope"]) if h["scl_slope"] else 1.0
    return a / sl, h


def load_echoes(d):
    fs = sorted(glob.glob(os.path.join(d, "vol_image001echo*.nii")))
    arrs, hdr = [], None
    for f in fs:
        a, h = read_nii(f)
        arrs.append(a)
        hdr = h
    return np.stack(arrs), hdr          # (echo, x, y, z)


def noise_sd(v, m):
    """면내 이웃 화소 차이로 잡음 SD 추정 (둘 다 마스크 안일 때만)."""
    d = np.diff(v, axis=0)
    mm = m[1:] & m[:-1]
    if mm.sum() < 50:
        return float("nan")
    return float(np.std(d[mm]) / np.sqrt(2))


print("=" * 78)
mag_d = os.path.join(DATA, "1", "1001_MREPT-NO1_20261001_01.nii")
phs_d = os.path.join(DATA, "1", "reconphs.nii")
mag, mh = load_echoes(mag_d)
phs, ph = load_echoes(phs_d)
print(f"magnitude {mag.shape}  dtype {mh.get_data_dtype()}  "
      f"scl_slope {mh['scl_slope']}  범위 {mag.min():.3g} ~ {mag.max():.3g}")
print(f"phase     {phs.shape}  dtype {ph.get_data_dtype()}  "
      f"scl_slope {ph['scl_slope']}  범위 {phs.min():.4g} ~ {phs.max():.4g}")

# 조직 / 공기 마스크 — magnitude 1에코 기준
m1 = mag[0]
thr = np.percentile(m1, 60)
tissue = ndimage.binary_erosion(m1 > thr, np.ones((3, 3, 1)), iterations=2)
air = ndimage.binary_erosion(m1 < np.percentile(m1, 12), np.ones((3, 3, 1)))
print(f"\n조직 {int(tissue.sum())}화소  공기 {int(air.sum())}화소  "
      f"(문턱 {thr:.1f})")

# ----------------------------------------------------- A. 범위와 분포
print("\n" + "=" * 78)
print("A. 값의 범위와 분포")
q = np.percentile(phs[0][tissue], [0.5, 2.5, 25, 50, 75, 97.5, 99.5])
print("  에코1 조직 분위수 0.5/2.5/25/50/75/97.5/99.5 = "
      + " ".join(f"{x:+.3f}" for x in q))
print(f"  조직 전체 범위 {phs[0][tissue].min():+.3f} ~ {phs[0][tissue].max():+.3f}")
print(f"  라디안 위상이면 대략 -pi~+pi (-3.142~+3.142) 안에 들어야 한다.")

# ------------------------------------------- B. 조직 잡음 vs 공기 잡음 (결정적)
print("\n" + "=" * 78)
print("B. 조직 잡음 vs 공기 잡음  ← 결정적 시험")
print("   송수신 위상이면 조직은 신호가 있으므로 잡음이 작고, 공기는 위상이")
print("   무작위라 SD 가 pi/sqrt(3)=1.81 에 가까워야 한다. 둘이 같으면 위상이 아니다.")
snr_rows = []
for e in range(phs.shape[0]):
    nt = noise_sd(phs[e], tissue)
    na = noise_sd(phs[e], air)
    sig = float(np.median(mag[e][tissue]))
    nmag = float(np.std(mag[e][air])) or float("nan")
    snr = sig / nmag * np.sqrt(2 - np.pi / 2) if nmag == nmag else float("nan")
    snr_rows.append((e + 1, sig, snr, nt, na, nt / na if na else float("nan")))
    print(f"   에코{e+1}  magnitude 조직중앙 {sig:8.1f}  SNR {snr:6.1f}  |  "
          f"위상잡음 조직 {nt:.4f}  공기 {na:.4f}  비 {nt/na:.3f}")
ratio = np.mean([r[5] for r in snr_rows])
ok_b = ratio < 0.5
print(f"\n   조직/공기 잡음비 평균 {ratio:.3f}  ->  "
      + ("위상으로 볼 수 있다" if ok_b else "**위상이 아니다** (26-08-20과 같은 문제)"))

# --------------------------------------------------- C. 에코 간 거동
print("\n" + "=" * 78)
print("C. 에코 간 거동 (스핀에코면 송수신 위상은 에코에 무관해야 한다)")
ref = phs[0]
for e in range(1, phs.shape[0]):
    d = phs[e] - ref                                # 이미 언래핑된 값으로 본다
    print(f"   에코{e+1}-에코1  조직 중앙 {np.median(d[tissue]):+.4f}  "
          f"IQR {np.percentile(d[tissue],75)-np.percentile(d[tissue],25):.4f}")
med = [float(np.median((phs[e] - ref)[tissue])) for e in range(phs.shape[0])]
slope = np.polyfit(np.arange(1, phs.shape[0] + 1), med, 1)[0]
print(f"   에코에 대한 기울기 {slope:+.4f} rad/echo "
      + ("(무관에 가깝다 -> 평균으로 SNR 이득 가능)" if abs(slope) < 0.05
         else "(선형 성분 있음 -> B0/eddy 기여)"))

# ----------------------------------------- D. 공간 구조 + 잡음 저감 가능량
print("\n" + "=" * 78)
print("D. 에코 평균으로 줄어드는 위상 잡음")
pmean = phs.mean(0)
nt1 = noise_sd(phs[0], tissue)
ntm = noise_sd(pmean, tissue)
print(f"   에코1 조직 위상잡음 {nt1:.4f} rad -> 8에코 평균 {ntm:.4f} rad "
      f"(개선 {nt1/ntm:.2f}배, 이론 {np.sqrt(8):.2f}배)")

# 그림
zc = mag.shape[3] // 2
fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for j, (t, v, cm, kw) in enumerate([
        ("magnitude 에코1", mag[0, :, :, zc], "gray", {}),
        ("phase 에코1", phs[0, :, :, zc], "twilight", dict(vmin=-np.pi, vmax=np.pi)),
        ("phase 에코8", phs[-1, :, :, zc], "twilight", dict(vmin=-np.pi, vmax=np.pi)),
        ("phase 8에코 평균", pmean[:, :, zc], "twilight",
         dict(vmin=-np.pi, vmax=np.pi))]):
    im = ax[0, j].imshow(v.T, cmap=cm, **kw)
    ax[0, j].set_title(t, fontsize=10)
    ax[0, j].set_xticks([]); ax[0, j].set_yticks([])
    fig.colorbar(im, ax=ax[0, j], fraction=0.046)
ax[1, 0].hist(phs[0][tissue], bins=120, color="C0")
ax[1, 0].set_title("에코1 조직 위상 분포", fontsize=10)
ax[1, 0].axvline(-np.pi, color="r", lw=0.8); ax[1, 0].axvline(np.pi, color="r", lw=0.8)
ax[1, 1].hist(phs[0][air], bins=120, color="C1")
ax[1, 1].set_title("에코1 공기 위상 분포", fontsize=10)
ax[1, 2].plot(range(1, phs.shape[0] + 1), [r[3] for r in snr_rows], "o-", label="조직")
ax[1, 2].plot(range(1, phs.shape[0] + 1), [r[4] for r in snr_rows], "s-", label="공기")
ax[1, 2].set_xlabel("에코"); ax[1, 2].set_ylabel("위상 잡음 SD (rad)")
ax[1, 2].legend(); ax[1, 2].grid(alpha=0.3)
ax[1, 3].plot(range(1, phs.shape[0] + 1), [r[2] for r in snr_rows], "o-")
ax[1, 3].set_xlabel("에코"); ax[1, 3].set_ylabel("magnitude SNR")
ax[1, 3].grid(alpha=0.3)
fig.suptitle("261001 폴더 1 — reconphs.nii 진단", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "reconphs_diag.png"), dpi=130, bbox_inches="tight")
print("\n저장:", os.path.join(OUT, "reconphs_diag.png"))
