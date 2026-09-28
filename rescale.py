# -*- coding: utf-8 -*-
"""
표시 스케일 재조정 — 뇌 기준 창(window) + 수신 코일 감도 불균일 보정
==================================================================

왜 필요한가
-----------
기존 그림의 회색 창은 `percentile(PRE[PRE>0], [2, 99.5])` 였다. 아래쪽 끝이
공기·두개골 수준(14)이라 계조의 아래 3분의 1이 뇌가 아닌 것에 쓰이고, 위쪽은
POST 뇌 화소의 16~22%가 잘려 흰색으로 날아갔다. 그래서 Gd 증강이 회색 변화로
잘 안 보였다.

여기에 더해 **수신 코일 감도 기울기**가 있다. 0922 폴더 4 gre axial z=8 에서
뇌 중앙값이 행 63 에서 379, 행 195 에서 213 으로 매끄럽게 1.8배 떨어진다.
이 기울기가 뇌 안 밝기 폭의 대부분을 만들기 때문에, 창만 좁히면 코일에 가까운
쪽이 먼저 포화된다.

무엇을 하는가
-------------
1. PRE 의 log 세기를 뇌 화소에서만 3차 다항식으로 적합해 매끄러운 감도장을
   만든다 (뇌 중앙값이 1이 되도록 정규화).
2. **PRE 와 POST 를 같은 장으로 나눈다.** 같은 수를 나누므로 (POST-PRE)/PRE 는
   그대로다. 실제로 최대 차이가 1e-13 %p 수준인 것을 스크립트가 매번 확인해
   출력한다. 즉 이 보정은 화면 표시만 바꾸고 정량값은 건드리지 않는다.
3. 보정된 PRE 의 뇌 화소 분위수로 창을 잡고, 그 창을 PRE·POST 에 똑같이 쓴다.

뇌 마스크
---------
`_focus_depth.py` 의 마스크(밝기 문턱 + 최대 연결성분)에 두 가지 문제가 있었다.

1. 측두근과 등쪽 두피 띠를 물고 들어왔다. 침식으로 얇은 연결을 끊고 가장 큰
   덩어리만 남긴 뒤 원래 범위 안에서 다시 키워(geodesic dilation) 끊는다.
   반경은 화소가 아니라 mm 로 준다 (`ERODE_MM`, 기본 1.0 mm). GRE 는 0.117,
   FSE 는 0.156 mm/화소라 화소로 고정하면 두 계열의 침식 세기가 달라진다.
2. 문턱 `PRE >= DARK_FRAC * 조직중앙값` 이 전역이라, 코일 감도 기울기가 있으면
   코일에서 먼 슬라이스의 뇌 전체가 문턱 아래로 내려간다. 0922 폴더 4 fse
   관상 슬라이스 0~3 의 마스크가 1680~2429화소까지 주저앉았다 (슬라이스 6 은
   17932). 그래서 마스크와 감도장을 번갈아 두 번 다시 구한다(`brain_and_field`).
   같은 슬라이스가 4989~11207화소로 회복된다.

아래·가쪽 경계는 여전히 완전하지 않다 (두께 1 mm 를 넘는 근육 접촉면은 침식이
끊지 못한다). 해부학적 분할이 아니라 창을 잡기 위한 근사다.

한계 (반드시 같이 읽을 것)
--------------------------
* 창을 좁히면 **잡음과 정합 잔차도 같이 커 보인다.** 증강처럼 보이는 것이
  늘어나므로, 판정은 계속 증강 % 와 `bbbo_metrics` 숫자로 한다.
* 3차 다항식의 공간 규모는 FOV 의 약 1/3 (30 mm FOV 에서 10 mm 정도)이다.
  1~3 mm 크기의 국소 병소는 이 장에 흡수되지 않지만, 한쪽 반구 전체에 걸친
  넓은 증강은 일부 깎일 수 있다. 넓은 변화를 보려면 보정 전 그림을 같이 본다.
* 감도장을 뇌에서만 적합하므로 뇌 밖(두피·근육)에서는 외삽이다. 보정 영상의
  두피 밝기는 해석하지 않는다.

실행
----
    python "_rescale.py"            # RAT 1, 4 비교 그림
    python "_rescale.py" 4          # RAT 4 만
    python "_rescale.py" 4 --full   # 선택한 옵션(CHOICE)으로 전체 슬라이스 출력
"""
import os
import re
import sys
import numpy as np
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

DARK_FRAC = 0.85        # 조직 중앙값 대비 이 배수 미만은 두개골·경막·배경
MIN_BRAIN_VOX = 5000
ERODE_MM = 1.0          # 측두근 연결을 끊는 침식 반경 (mm). 화소 크기로 환산한다
POLY_DEG = 3            # 감도장 다항식 차수
MIN_SPAN_SD = 8.0       # 창 폭의 하한을 PRE 잡음 SD 의 배수로 준다
CHOICE = "B"            # --full 에서 쓸 옵션: 현재 / A / B / C / D
ENH_VMAX = 60


# ----------------------------------------------------------------- 뇌 마스크
def _disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return (y * y + x * x <= r * r)[None, :, :]


def brain_mask(pre, tissue, r):
    med = float(np.median(pre[tissue]))
    b = ndimage.binary_opening(tissue & (pre >= DARK_FRAC * med), np.ones((1, 3, 3)))
    b = np.stack([ndimage.binary_fill_holes(b[z]) for z in range(b.shape[0])])
    er = ndimage.binary_erosion(b, _disk(r))
    lab, n = ndimage.label(er)
    if n == 0:
        return None
    sizes = ndimage.sum(er, lab, range(1, n + 1))
    if sizes.max() < MIN_BRAIN_VOX / 4:
        return None
    cur = lab == int(np.argmax(sizes)) + 1
    for _ in range(r + 4):                       # 원래 범위 안에서만 복원
        cur = ndimage.binary_dilation(cur, np.ones((1, 3, 3))) & b
    return np.stack([ndimage.binary_fill_holes(cur[z]) for z in range(cur.shape[0])])


# --------------------------------------------------------------- 감도 불균일
def _monomials(z, y, x, deg):
    cols = []
    for i in range(deg + 1):
        for j in range(deg + 1 - i):
            for k in range(deg + 1 - i - j):
                cols.append((z ** i) * (y ** j) * (x ** k))
    return np.stack(cols, 1)


def bias_field(vol, mask, deg=POLY_DEG):
    """뇌 화소의 log 세기를 다항식으로 적합한 매끄러운 배수장 (뇌 중앙값 = 1)."""
    nz, ny, nx = vol.shape
    gz, gy, gx = np.meshgrid(np.linspace(-1, 1, nz), np.linspace(-1, 1, ny),
                             np.linspace(-1, 1, nx), indexing="ij")
    A = _monomials(gz[mask], gy[mask], gx[mask], deg)
    coef, *_ = np.linalg.lstsq(A, np.log(np.clip(vol[mask], 1.0, None)), rcond=None)
    fld = np.exp((_monomials(gz.ravel(), gy.ravel(), gx.ravel(), deg) @ coef)
                 .reshape(vol.shape))
    return fld / float(np.median(fld[mask]))


def brain_and_field(pre, tissue, px_mm, iters=2):
    """뇌 마스크와 감도장을 번갈아 다시 구한다.

    `PRE >= DARK_FRAC * 조직중앙값` 은 전역 문턱이라 코일 감도 기울기가 있으면
    코일에서 먼 슬라이스의 뇌 전체가 문턱 아래로 내려간다. 0922 폴더 4 fse
    관상에서 슬라이스 0~3 의 마스크가 1680~2429화소까지 주저앉았다(슬라이스 6
    은 17932). 그래서 거친 마스크로 감도장을 한 번 적합해 나눈 뒤, 평평해진
    영상에서 마스크를 다시 잡고 그 마스크로 감도장을 다시 적합한다.
    """
    r = max(1, int(round(ERODE_MM / px_mm)))
    br = brain_mask(pre, tissue, r)
    if br is None:
        return None, None
    for _ in range(iters):
        nb = brain_mask(pre / bias_field(pre, br), tissue, r)
        if nb is None:
            break
        br = nb
    return br, bias_field(pre, br)


def noise_sd(vol, brain):
    """면내 이웃 화소 차이로 뇌 안 잡음 SD 를 추정한다 (둘 다 뇌 안일 때만)."""
    d = np.diff(vol, axis=2)
    m = brain[:, :, 1:] & brain[:, :, :-1]
    return float(np.std(d[m]) / np.sqrt(2))


def _widen(wl, med, floor):
    """창이 너무 좁으면 중앙값의 상대 위치를 유지한 채 최소 폭까지 넓힌다.

    분위수만으로 창을 잡으면 조직 대비가 작고 잡음이 큰 계열에서 창이 잡음 폭까지
    내려앉는다. 0922 폴더 4 fse 축상이 그랬다 (창 371, PRE 잡음 SD 73.6 = 계조의
    19.9%, POST 25.1%). 잡음 ±1 SD 가 계조의 4분의 1을 덮으면 화면이 거칠어진다.
    """
    lo, hi = float(wl[0]), float(wl[1])
    span = hi - lo
    if span >= floor:
        return lo, hi
    f = (med - lo) / span
    lo2 = med - f * floor
    return lo2, lo2 + floor


def options(pre, post, brain, fld):
    """(이름, 표시용 PRE, 표시용 POST, (vmin, vmax)) 목록."""
    preC, postC = pre / fld, post / fld
    med = float(np.median(preC[brain]))
    floor = MIN_SPAN_SD * noise_sd(preC, brain)
    return {
        "현재": ("현재 (전체 2~99.5%)", pre, post,
                 tuple(np.percentile(pre[pre > 0], [2, 99.5]))),
        "A": ("A 뇌 기준 창 (보정 없음)", pre, post,
              tuple(np.percentile(pre[brain], [2, 98]))),
        "B": ("B 불균일 보정 + 뇌 2~98%", preC, postC,
              _widen(np.percentile(preC[brain], [2, 98]), med, floor)),
        "C": ("C 불균일 보정 + 뇌 5~95%", preC, postC,
              _widen(np.percentile(preC[brain], [5, 95]), med, floor)),
        # D 는 위쪽 끝을 POST 분위수로 잡는다. 포화되는 뇌 화소가 계열과 무관하게
        # 정확히 (100-85)% 로 고정되므로 계열끼리 겉보기 밝기를 비교할 수 있다.
        "D": ("D 불균일 보정 + PRE 2% ~ POST 85%", preC, postC,
              (float(np.percentile(preC[brain], 2)),
               float(np.percentile(postC[brain], 85)))),
    }


def _gray(v, brain, wl):
    return (np.median(v[brain]) - wl[0]) / (wl[1] - wl[0]) * 100


def _clip(v, brain, wl):
    return float(((v[brain] > wl[1]) | (v[brain] < wl[0])).mean() * 100)


# ------------------------------------------------------------------ 본 처리
def series(rat):
    d = np.load(os.path.join(BASE, f"RAT {rat}", "corrected_volumes.npz"))
    for kind in ("gre", "fse"):
        for ori in ("axial", "coronal"):
            k = f"{kind}_{ori}_PRE"
            if k not in d.files:
                continue
            posts = sorted([x for x in d.files if x.startswith(f"{kind}_{ori}_POST")],
                           key=lambda x: int(re.search(r"POST(\d*)$", x).group(1) or 1))
            if not posts:
                continue
            yield (kind, ori, d[k].astype(float), d[posts[-1]].astype(float),
                   posts[-1].split("_")[-1], d[f"{kind}_{ori}_analysis_mask"],
                   d[f"{kind}_{ori}_spacing"])


def run(rat, full=False):
    out = os.path.join(BASE, f"RAT {rat}", "rescaled" if full else "scale_test")
    os.makedirs(out, exist_ok=True)
    log = []
    for kind, ori, pre, post, pname, tis, sp in series(rat):
        br, fld = brain_and_field(pre, tis, abs(float(sp[0])))
        if br is None:
            print(f"  [경고] {kind} {ori}: 뇌 마스크 실패")
            continue
        vox = abs(float(sp[0] * sp[1] * sp[2]))
        OPT = options(pre, post, br, fld)
        per_z = " ".join(str(int(br[z].sum())) for z in range(pre.shape[0]))
        head = (f"### RAT {rat} {kind} {ori} (PRE vs {pname}) — 뇌 마스크 "
                f"{br.sum() * vox:.0f} mm3, 감도장 {fld[br].min():.2f}~{fld[br].max():.2f}배"
                f"\n  슬라이스별 뇌 화소: {per_z}")
        log.append(head)
        print(head)

        # 보정이 증강 %를 바꾸지 않음을 매번 확인
        _, pB, qB, _ = OPT["B"]
        e0 = (post[br] - pre[br]) / pre[br] * 100
        e1 = (qB[br] - pB[br]) / pB[br] * 100
        chk = f"  [검증] 증강 % 최대 차이 {np.abs(e0 - e1).max():.2e} %p (0 이어야 함)"
        log.append(chk)
        print(chk)
        for key in ("현재", "A", "B", "C", "D"):
            nm, vp, vq, wl = OPT[key]
            s = (f"  {nm}: 창 {wl[0]:.0f}~{wl[1]:.0f} | 뇌 중앙 회색 "
                 f"{_gray(vp, br, wl):.0f}% -> {_gray(vq, br, wl):.0f}% "
                 f"(d{_gray(vq, br, wl) - _gray(vp, br, wl):.1f}%p) | 창밖 뇌화소 "
                 f"{_clip(vp, br, wl):.1f}/{_clip(vq, br, wl):.1f}%")
            log.append(s)
            print(s)

        idx = np.argwhere(br.any(0))
        r0, c0 = np.maximum(idx.min(0) - 6, 0)
        r1, c1 = idx.max(0) + 7
        zz = np.argwhere(br.sum((1, 2)) > 200).ravel()      # 뇌가 실제로 있는 슬라이스
        zs = (range(pre.shape[0]) if full
              else sorted({int(v) for v in
                           np.linspace(zz[0], zz[-1], min(4, len(zz)))}))

        for z in zs:
            keys = [CHOICE] if full else ["현재", "A", "B", "C", "D"]
            fig, ax = plt.subplots(len(keys), 3, figsize=(11.4, 3.7 * len(keys)),
                                   squeeze=False)
            for i, key in enumerate(keys):
                nm, vp, vq, wl = OPT[key]
                enh = np.where(br[z], (vq[z] - vp[z]) / (vp[z] + 1e-6) * 100, np.nan)
                ax[i, 0].imshow(vp[z, r0:r1, c0:c1], cmap="gray", vmin=wl[0], vmax=wl[1])
                ax[i, 1].imshow(vq[z, r0:r1, c0:c1], cmap="gray", vmin=wl[0], vmax=wl[1])
                im = ax[i, 2].imshow(enh[r0:r1, c0:c1], cmap="jet", vmin=0, vmax=ENH_VMAX)
                ax[i, 0].set_ylabel(nm, fontsize=9)
                if i == 0:
                    ax[i, 0].set_title("PRE", fontsize=10)
                    ax[i, 1].set_title(pname, fontsize=10)
                    ax[i, 2].set_title("증강 % (뇌만)", fontsize=10)
                ax[i, 1].set_xlabel(
                    f"창 {wl[0]:.0f}~{wl[1]:.0f}  |  뇌 중앙 회색 "
                    f"{_gray(vp, br, wl):.0f}%→{_gray(vq, br, wl):.0f}%  |  창밖 "
                    f"{_clip(vp, br, wl):.1f}/{_clip(vq, br, wl):.1f}%", fontsize=7.5)
                for a in ax[i]:
                    a.set_xticks([])
                    a.set_yticks([])
            fig.colorbar(im, ax=ax[:, 2].tolist(), fraction=0.03, pad=0.02, label="% 증강")
            fig.suptitle(f"RAT {rat} {kind} {ori}  slice {z} (0-index)"
                         + ("" if full else " — 표시 스케일 비교\n"
                            "증강 % 는 다섯 경우 모두 같다 (창은 화면 표시만 바꾼다)"),
                         fontsize=11)
            fig.savefig(os.path.join(out, f"{'rescaled' if full else 'scale'}"
                                     f"_{kind}_{ori}_slice{z:02d}.png"),
                        dpi=130, bbox_inches="tight")
            plt.close(fig)

        # 감도장 자체도 저장 (무엇을 나눴는지 눈으로 확인)
        if not full:
            n = min(4, pre.shape[0])
            fig, ax = plt.subplots(1, n, figsize=(3.2 * n, 3.4), squeeze=False)
            for a, z in zip(ax[0], np.linspace(0, pre.shape[0] - 1, n).astype(int)):
                im = a.imshow(np.where(br[z], fld[z], np.nan), cmap="viridis",
                              vmin=fld[br].min(), vmax=fld[br].max())
                a.set_title(f"slice {z}", fontsize=9)
                a.set_xticks([])
                a.set_yticks([])
            fig.colorbar(im, ax=ax[0].tolist(), fraction=0.03, label="배수")
            fig.suptitle(f"RAT {rat} {kind} {ori} — 적합된 코일 감도장", fontsize=11)
            fig.savefig(os.path.join(out, f"biasfield_{kind}_{ori}.png"),
                        dpi=120, bbox_inches="tight")
            plt.close(fig)

    with open(os.path.join(out, "scale_report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    print("저장:", out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    full = "--full" in sys.argv
    for r in (args or ["1", "4"]):
        run(r, full)


# ------------------------------------------- 개별 프레임(무축 bare) 다시 뽑기
def frames(rat, kinds=("gre",), oris=("axial", "coronal"), key=None, dpi=140):
    """`RAT n/{kind}/{ori}/individual` 과 같은 형식의 낱장 그림을 선택한 창으로 다시 만든다.

    출력은 `individual_rescaled/` 로 나가고 파일 이름은 원본과 같다. 원본은 건드리지
    않으므로 두 폴더를 나란히 열어 비교할 수 있다.
    """
    key = key or CHOICE
    d = np.load(os.path.join(BASE, f"RAT {rat}", "corrected_volumes.npz"))
    date = os.path.basename(os.path.dirname(BASE)).replace("-", "")[2:]   # 26-09-22 -> 0922
    for kind in kinds:
        for ori in oris:
            k = f"{kind}_{ori}_PRE"
            if k not in d.files:
                continue
            pre = d[k].astype(float)
            tis = d[f"{kind}_{ori}_analysis_mask"]
            sp = d[f"{kind}_{ori}_spacing"]
            br, fld = brain_and_field(pre, tis, abs(float(sp[0])))
            if br is None:
                print(f"  [경고] {kind} {ori}: 뇌 마스크 실패")
                continue
            preC = pre / fld
            if key in ("현재", "A"):        # 보정을 쓰지 않는 창
                fld = np.ones_like(pre)
                preC = pre
            posts = sorted([x for x in d.files if x.startswith(f"{kind}_{ori}_POST")],
                           key=lambda x: int(re.search(r"POST(\d*)$", x).group(1) or 1))
            wl = options(pre, d[posts[-1]].astype(float), br, fld)[key][3]

            out = os.path.join(BASE, f"RAT {rat}", kind, ori, "individual_rescaled")
            os.makedirs(out, exist_ok=True)
            tag = f"{date}_F{rat}_{kind}_{ori}"
            vols = [("preGd", preC)] + [(f"post{i + 1}", d[p].astype(float) / fld)
                                        for i, p in enumerate(posts)]
            for z in range(pre.shape[0]):
                for nm, v in vols:
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(v[z], cmap="gray", vmin=wl[0], vmax=wl[1])
                    ax.axis("off")
                    fig.savefig(os.path.join(out, f"{tag}_slice{z}_{nm}.png"),
                                dpi=dpi, bbox_inches="tight", pad_inches=0)
                    plt.close(fig)
            print(f"  {kind} {ori}: {len(vols) * pre.shape[0]}장 -> {out}"
                  f"  (옵션 {key}, 창 {wl[0]:.0f}~{wl[1]:.0f})")
