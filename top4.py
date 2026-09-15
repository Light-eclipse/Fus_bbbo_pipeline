# -*- coding: utf-8 -*-
"""개체당 표적이 4개였다는 설계에 맞춰 **상위 4개 병변**을 뽑고 서로 간 거리를 낸다.

`_lesion_summary.py`가 만든 병변 목록을 증거 강도로 정렬해 개체별 상위 4개를 고른다.
정렬 기준은 (1) 독립 획득 재현 수, (2) 인공물 표시 없음, (3) 최대 부피 순이다.

**주의.** 표적이 4개였다는 사실이 상위 4개를 진짜로 만들어 주지 않는다. 검출된
후보가 표적 수보다 많으면 그중 일부는 인공물이고, 적으면 일부 표적은 열리지 않았거나
검출 문턱 아래다. 여기서 하는 일은 "4개라면 이 넷이 가장 근거가 강하다"를 보여주는
것뿐이고, 판정은 재현 획득 수와 표시를 보고 해야 한다.

병변 간 거리는 자기 점검용이다. 표적을 일정 간격으로 잡았다면 검출된 병변 간 거리가
그 간격과 맞아야 한다. 맞지 않으면 일부가 인공물이거나 표적이 빠진 것이다.

실행: python "_top4.py"
"""
import os
import csv
import itertools
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
N_TARGET = 4

rows = list(csv.DictReader(open(os.path.join(BASE, "lesion_summary.csv"),
                                encoding="utf-8-sig")))
out_rows, dist_rows = [], []

for rat in ("1", "3", "4"):
    mine = [x for x in rows if x["rat_folder"] == rat]
    if not mine:
        continue

    def score(x):
        clean = 0 if "(" in x["confidence"] else 1      # 인공물 표시 없으면 우선
        return (-int(x["n_independent_acquisitions"]), -clean, -float(x["max_volume_mm3"]))

    mine.sort(key=score)
    top = mine[:N_TARGET]
    print("=" * 80)
    print(f"폴더 {rat} — 표적 4개 가정, 근거 강한 순 상위 {len(top)}개 "
          f"(전체 검출 병변 {len(mine)}개)")
    print(f"  {'순위':4}{'병변':4}{'부피mm3':>9}{'깊이mm':>8}{'순누출%p':>9}{'획득':>4}  "
          f"{'조합':36}{'판정'}")
    P = []
    for i, x in enumerate(top, 1):
        p = np.array([float(v) for v in x["patient_xyz_mm"].split()])
        P.append(p)
        print(f"  {i:<4}{x['lesion']:<4}{x['max_volume_mm3']:>9}"
              f"{x['depth_below_brain_surface_mm']:>8}{x['mean_net_leakage_pp']:>9}"
              f"{x['n_independent_acquisitions']:>4}  {x['acquisitions']:36}"
              f"{x['confidence']}")
        out_rows.append([rat, i, x["lesion"], x["max_volume_mm3"],
                         x["depth_below_brain_surface_mm"], x["mean_net_leakage_pp"],
                         x["n_independent_acquisitions"], x["acquisitions"],
                         x["confidence"], x["patient_xyz_mm"]])
    if len(P) >= 2:
        print("  병변 간 거리 (mm):")
        for (i, a), (j, b) in itertools.combinations(list(enumerate(P, 1)), 2):
            dd = float(np.linalg.norm(a - b))
            print(f"    순위 {i} - 순위 {j}: {dd:6.2f}")
            dist_rows.append([rat, i, j, round(dd, 2)])

with open(os.path.join(BASE, "top4_lesions.csv"), "w",
          newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["rat_folder", "rank", "lesion_id", "max_volume_mm3",
                "depth_below_brain_surface_mm", "mean_net_leakage_pp",
                "n_independent_acquisitions", "acquisitions", "confidence",
                "patient_xyz_mm"])
    w.writerows(out_rows)
with open(os.path.join(BASE, "top4_pairwise_distance.csv"), "w",
          newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["rat_folder", "rank_a", "rank_b", "distance_mm"])
    w.writerows(dist_rows)
print("\n저장: top4_lesions.csv, top4_pairwise_distance.csv")
