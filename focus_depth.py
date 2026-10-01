# -*- coding: utf-8 -*-
"""검출된 병소의 **뇌 표면 아래 깊이**를 mm로 잰다.

표적 깊이가 이번 실험의 질문이므로 병소가 뇌 표면에서 얼마나 들어가 있는지가 핵심
수치다.

깊이 정의
---------
뇌 마스크를 만들고 **거리변환**으로 잰다. 즉 깊이 = 병소 중심에서 가장 가까운 뇌
경계까지의 거리. 화소 이방성(면내 0.117~0.156 mm, 슬라이스 1.0 mm)은 거리변환의
sampling 인자로 처리한다.

처음에는 병소가 있는 열을 따라 위에서 내려오며 두개골 암선을 찾는 방식을 썼는데,
암선 탐색이 실패하는 열이 있어서(최소점이 조직 시작 행과 같아짐) 깊이가 과대평가됐다.
거리변환은 방향을 가정하지 않고 열별 실패도 없다. 대신 "등쪽에서의 깊이"가 아니라
"가장 가까운 뇌 경계까지의 거리"라는 점은 해석할 때 감안해야 한다. 병소가 뇌 옆면에
가까우면 옆면까지의 거리가 나온다.

뇌 마스크 만드는 법
------------------
1. 조직 마스크 안에서 두개골·경막에 해당하는 어두운 화소를 뺀다
   (PRE가 조직 중앙값의 `DARK_FRAC` 배 미만).
2. 남은 것 중 **부피가 가장 큰 3차원 연결성분**을 뇌로 본다. 두개골 암선이 뇌와
   두피를 갈라 주므로 이 성분이 뇌가 된다.
3. 작은 구멍은 메운다.

이 마스크는 근사다. 반드시 `brain_mask_check/`에 저장되는 오버레이 그림으로 한 번
확인하고 쓸 것. 암선이 끊긴 슬라이스에서는 두피가 뇌에 붙어 깊이가 과대평가된다.

실행
----
    python "_focus_depth.py"          # 폴더 1, 4 전부
    python "_focus_depth.py" 4        # 폴더 4만
"""
import os
import sys
import csv
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

DARK_FRAC = 0.85        # 조직 중앙값 대비 이 배수 미만이면 두개골·경막·배경으로 본다
MIN_BRAIN_VOX = 5000    # 뇌 성분 최소 화소수

RATS = sys.argv[1:] or ["1", "2", "3"]


# v0.8(`_rescale.py`)에서 고친 마스크를 그대로 쓴다. 기존 마스크의 문제 두 가지:
#   1. 침식 반경이 화소 단위였다. GRE 0.137 / FSE 0.182 mm/화소라 계열마다 세기가
#      달랐다. mm 로 준다.
#   2. 문턱 `PRE >= DARK_FRAC * 조직중앙값` 이 전역이라 코일 감도 기울기가 있으면
#      코일에서 먼 슬라이스의 뇌 전체가 문턱 아래로 떨어진다. 마스크와 감도장을
#      번갈아 다시 구해 푼다.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_rescale", os.path.join(BASE, "_rescale.py"))
_rs = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_rs)


def brain_mask(pre, tissue, px_mm):
    br, _ = _rs.brain_and_field(pre, tissue, px_mm)
    return br


rows_out = []
for rat in RATS:
    out_dir = os.path.join(BASE, f"RAT {rat}")
    npz = os.path.join(out_dir, "corrected_volumes.npz")
    if not os.path.exists(npz):
        print(f"[건너뜀] {npz} 없음", flush=True)
        continue
    d = np.load(npz)
    chk = os.path.join(out_dir, "brain_mask_check")
    os.makedirs(chk, exist_ok=True)
    print("=" * 76)
    print(f"폴더 {rat}")

    DEPTH, BRAIN = {}, {}
    for kind in ("gre", "fse"):
        for ori in ("axial", "coronal"):
            k = f"{kind}_{ori}_PRE"
            if k not in d.files:
                continue
            pre = d[k].astype(float)
            tis = d[f"{kind}_{ori}_analysis_mask"]
            sp = d[f"{kind}_{ori}_spacing"]      # (sx, sy, sz)
            br = brain_mask(pre, tis, abs(float(sp[0])))
            if br is None:
                print(f"  [경고] {kind} {ori}: 뇌 마스크 실패")
                continue
            # sampling 은 배열 축 순서 (z, y, x)
            dist = ndimage.distance_transform_edt(
                br, sampling=(float(sp[2]), float(sp[1]), float(sp[0])))
            DEPTH[(kind, ori)] = dist
            BRAIN[(kind, ori)] = br
            print(f"  {kind} {ori}: 뇌 마스크 {int(br.sum())}화소 "
                  f"({br.sum()*abs(sp[0]*sp[1]*sp[2]):.0f} mm3), "
                  f"최대 깊이 {dist.max():.2f} mm")
            # 확인용 오버레이
            nz = pre.shape[0]
            cols = min(7, nz); rr = int(np.ceil(nz / cols))
            fig, axes = plt.subplots(rr, cols, figsize=(2.4 * cols, 2.6 * rr), squeeze=False)
            axes = axes.ravel()
            wl = np.percentile(pre[pre > 0], [2, 99])
            for z in range(nz):
                axes[z].imshow(pre[z], cmap="gray", vmin=wl[0], vmax=wl[1])
                axes[z].contour(br[z], levels=[0.5], colors="lime", linewidths=0.8)
                axes[z].set_title(f"slice {z}", fontsize=8)
            for a in axes:
                a.axis("off")
            fig.suptitle(f"폴더 {rat} {kind} {ori} — 뇌 마스크 확인 (초록 윤곽)", fontsize=12)
            fig.tight_layout()
            fig.savefig(os.path.join(chk, f"brainmask_{kind}_{ori}.png"),
                        dpi=110, bbox_inches="tight")
            plt.close(fig)

    for which, fname in (("국소", "local_focus/local_focus.csv"),
                         ("거울", "lateral_focus/lateral_focus.csv")):
        p = os.path.join(out_dir, fname)
        if not os.path.exists(p):
            continue
        for x in csv.DictReader(open(p, encoding="utf-8-sig")):
            kind, ori = x["kind"], x["orientation"]
            if (kind, ori) not in DEPTH:
                continue
            dist = DEPTH[(kind, ori)]
            br = BRAIN[(kind, ori)]
            z = int(round(float(x["center_z"])))
            r = int(x["center_row"]); c = int(x["center_col"])
            if not (0 <= z < dist.shape[0] and 0 <= r < dist.shape[1]
                    and 0 <= c < dist.shape[2]):
                continue
            inside = bool(br[z, r, c])
            # 중심이 마스크 밖이면(경계 걸침) 3x3x3 이웃의 최대 깊이를 쓴다
            sub = dist[max(0, z-1):z+2, max(0, r-2):r+3, max(0, c-2):c+3]
            dep = float(dist[z, r, c]) if inside else float(sub.max())
            ctrl = x.get("shell_enh_last_pct", x.get("mirror_enh_last_pct", ""))
            flag = "뇌 안" if inside else ("뇌 경계/밖" if dep > 0 else "뇌 밖")
            print(f"  [{which}] {kind:3} {ori:8} #{x['rank']}  {x['volume_mm3']:>6} mm3  "
                  f"순 누출 {x['net_leakage_pp_lastPOST']:>6} %p  대조 {ctrl:>6} %  "
                  f"가장자리 {x['edge_slice']:4} {flag:9} "
                  f"-> 뇌 표면 아래 **{dep:.2f} mm**")
            rows_out.append([rat, which, kind, ori, x["rank"], x["volume_mm3"],
                             x["net_leakage_pp_lastPOST"], ctrl, x["edge_slice"],
                             x["slices"], z, r, c, "예" if inside else "아니오",
                             round(dep, 2)])

if rows_out:
    out = os.path.join(BASE, "focus_depth.csv")
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rat_folder", "detector", "kind", "orientation", "rank",
                    "volume_mm3", "net_leakage_pp", "control_enh_pct", "edge_slice",
                    "slices", "center_z", "center_row", "center_col",
                    "center_inside_brain_mask", "depth_below_brain_surface_mm"])
        w.writerows(rows_out)
    print("\n저장:", os.path.basename(out))
else:
    print("\n깊이를 잴 병소 후보가 없다.")
