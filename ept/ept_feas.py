# -*- coding: utf-8 -*-
"""261001 폴더 1 — phase EPT 로 sigma 를 어디까지 잴 수 있는가.

앞 단계에서 확정된 것
  * `reconphs` 는 접힘이 아니다 (조직 분포가 +-9 에 걸친 단봉 종 모양).
  * 실재하는 매끄러운 위상장이 있다 (3화소 평활 후 반쪽 상관 0.9965).
    장의 SD 는 약 2.6 rad, 폭은 +-9 rad.
  * 화소 잡음이 에코당 2.06 rad 다. magnitude SNR 34.3 이면 0.029 rad 여야
    하므로 **70배** 크다. 이건 물리가 아니라 내보내기/재구성 쪽 문제다.
  * 에코에 대한 기울기는 평활 후 0.10 rad/echo 로 작다. B0 는 대체로 재수렴한다.

여기서 계산하는 것
  1. sigma 0.7 S/m 가 만드는 위상이 얼마인가 (목표 신호의 크기)
  2. ROI 지름별로 sigma 를 몇 S/m 정확도로 잴 수 있는가 (잡음 한계)
  3. 지금 잡음 / 이론 잡음 두 경우를 나란히 — 내보내기를 고치면 얼마나 좋아지는가

방법: 반경 R 원판 안에서 2차 다항식 phi = a + bx + cy + d(x^2+y^2)/2 + ... 를
최소제곱으로 맞추면 라플라시안은 계수 조합으로 나온다. 화소 잡음이 독립이면
그 계수의 표준오차는 설계행렬에서 바로 나오므로 sigma 의 불확실도도 바로 나온다.
sigma = lap(phi) / (2 mu0 omega), 2 mu0 omega = 6312.2 (399.72 MHz).

실행: python "_ept_feas_1001.py"
"""
import numpy as np

PX_MM = 0.273
SLICE_MM = 1.0
GAMMA_HZ = 399.72e6
TWO_MU0_W = 2 * (4e-7 * np.pi) * 2 * np.pi * GAMMA_HZ

SIG_PHI_NOW = 2.06          # 실측, 에코당 (rad)
N_ECHO = 8
SIG_PHI_AVG = SIG_PHI_NOW / np.sqrt(N_ECHO)
SNR_MAG = 34.3
SIG_PHI_IDEAL = 1.0 / SNR_MAG

print("=" * 78)
print("1. 목표 신호의 크기")
for s in (0.5, 0.7, 1.0):
    lap = s * TWO_MU0_W
    for L_mm in (5, 10, 20):
        L = L_mm * 1e-3
        swing = lap * L ** 2 / 8
        print(f"   sigma {s:.1f} S/m -> lap(phi) {lap:7.0f} rad/m^2, "
              f"지름 {L_mm:2d} mm 에서 위상 폭 {swing*1e3:6.1f} mrad")
    print()

print("=" * 78)
print("2. ROI 지름별 sigma 불확실도")
print("   원판 안 2차 다항식 적합. 화소 0.273 x 0.273 mm, 슬라이스 1매.")
print(f"   {'지름(mm)':>9}{'화소수':>8}{'현재 잡음':>12}{'8에코 평균':>12}{'이론 잡음':>12}")
print(f"   {'':>9}{'':>8}{'(2.06 rad)':>12}{'(0.73 rad)':>12}{'(0.029 rad)':>12}")


def sigma_se(diam_mm, sig_phi, thick_slices=1):
    """원판 ROI 2차 적합에서 sigma 의 표준오차 (S/m)."""
    R = diam_mm / 2.0 / PX_MM
    n = int(np.ceil(R))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1].astype(float)
    m = (xx ** 2 + yy ** 2) <= R ** 2
    x = xx[m] * PX_MM * 1e-3
    y = yy[m] * PX_MM * 1e-3
    if x.size < 8:
        return np.nan, x.size
    # phi = c0 + c1 x + c2 y + c3 x^2 + c4 y^2 + c5 xy
    A = np.stack([np.ones_like(x), x, y, x ** 2, y ** 2, x * y], 1)
    cov = np.linalg.inv(A.T @ A) * sig_phi ** 2 / max(thick_slices, 1)
    # lap = 2*c3 + 2*c4  (면내 2D)
    var = 4 * (cov[3, 3] + cov[4, 4] + 2 * cov[3, 4])
    return np.sqrt(var) / TWO_MU0_W, x.size * max(thick_slices, 1)


for d_mm in (2, 3.5, 5, 7.5, 10, 15):
    s_now, npx = sigma_se(d_mm, SIG_PHI_NOW)
    s_avg, _ = sigma_se(d_mm, SIG_PHI_AVG)
    s_id, _ = sigma_se(d_mm, SIG_PHI_IDEAL)
    print(f"   {d_mm:>9.1f}{npx:>8}{s_now:>12.2f}{s_avg:>12.2f}{s_id:>12.3f}")

print("\n   여러 슬라이스를 묶으면 더 줄어든다 (같은 2차 적합, 화소만 늘어남)")
print(f"   {'지름(mm)':>9}{'슬라이스':>9}{'현재':>10}{'8에코':>10}{'이론':>10}")
for d_mm, ns in ((10, 1), (10, 5), (10, 14), (15, 14)):
    a, _ = sigma_se(d_mm, SIG_PHI_NOW, ns)
    b, _ = sigma_se(d_mm, SIG_PHI_AVG, ns)
    c, _ = sigma_se(d_mm, SIG_PHI_IDEAL, ns)
    print(f"   {d_mm:>9.1f}{ns:>9}{a:>10.2f}{b:>10.2f}{c:>10.3f}")

print("\n" + "=" * 78)
print("3. 판정")
s10, _ = sigma_se(10, SIG_PHI_AVG, 14)
s10i, _ = sigma_se(10, SIG_PHI_IDEAL, 14)
print(f"   지름 10 mm, 14 슬라이스, 8에코 평균 기준")
print(f"     지금 데이터     sigma 불확실도 {s10:.2f} S/m   목표 0.7 S/m 의 {s10/0.7:.0f}배")
print(f"     잡음만 고치면   sigma 불확실도 {s10i:.3f} S/m  목표의 {s10i/0.7*100:.0f}%")
print()
print("   즉 잡음은 내보내기를 고치면 넘을 수 있다. 남는 문제는 따로 있다.")
print(f"   관측된 매끄러운 위상장의 폭이 +-9 rad 인데, sigma 0.7 S/m 가 지름 20 mm")
print(f"   에서 만드는 폭은 {0.7*TWO_MU0_W*(0.02**2)/8*1e3:.0f} mrad 다. 약 "
      f"{9/(0.7*TWO_MU0_W*(0.02**2)/8):.0f}배 차이다.")
print("   위상장의 대부분은 전도도가 아니라 코일·정재파 성분이다. 라플라시안이")
print("   그 성분을 완전히 지워 주지 않으면(조화함수가 아니면) 남은 항이 전도도")
print("   신호보다 크다. 실제로 절편에 라플라시안을 걸면 조직의 49%에서 sigma 가")
print("   음수로 나온다. 스케일 오류로는 설명되지 않는다.")
