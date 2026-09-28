# -*- coding: utf-8 -*-
"""이 저장소의 이력을 만드는 스크립트.

저장소는 날짜별 분석 폴더에 흩어져 있던 스크립트를 모아 만든 것이다. 각 태그는
그 방법이 실제 데이터에 처음 쓰인 날짜의 파이프라인 상태다. `VERSIONS` 가 그
대응표이고, 파일은 원본을 **고치지 않고 그대로 복사**한다. 이름만 정리한다
(앞의 밑줄과 날짜를 뗀다). 저장소가 `무엇을 했는가`의 기록이기 때문이다.

두 가지 모드
-----------
    python tools/build_history.py
        `VERSIONS` 중 아직 태그가 없는 항목만 기존 이력 **위에 얹는다.**
        평소에 쓰는 모드다. 이미 올린 커밋은 건드리지 않는다.

    python tools/build_history.py --rebuild --force
        저장소를 비우고 v0.1부터 전부 다시 쌓는다. **커밋 SHA가 전부 바뀌므로
        이미 푸시했다면 강제 푸시가 필요하다.** 아직 푸시하지 않았거나 이력
        자체를 고쳐야 할 때만 쓴다. `--force` 없이는 실행되지 않는다.

다음 버전 올리는 법
------------------
1. 새 날짜 분석을 끝낸다.
2. `VERSIONS` 맨 뒤에 항목을 추가한다.
   (태그, 제목, 날짜, [(원본경로, 저장소경로)], 변경 요약 줄들)
   요약에는 **무엇이 틀렸었는지와 그것을 가른 숫자**를 적는다. 원본 스크립트
   주석에 근거가 없는 내용은 적지 않는다.
   날짜별 드라이버는 같은 파일의 연속 개정판이므로 저장소경로를 계속
   `run_dce.py` 로 둔다. 새로 생긴 스크립트만 파일로 추가한다.
3. `python tools/build_history.py` 를 돌린다.
4. `git push && git push --tags`

원본 경로(`ROOT`, `CODE`)는 이 연구 폴더의 실제 경로다. 다른 곳에서는 동작하지
않는다. 저장소의 다른 스크립트도 같은 이유로 절대 경로를 그대로 두었다.
"""
import os
import shutil
import subprocess
import sys

ROOT = r"C:\Users\user\Desktop\대학원\6. 원자력의학원"
REPO = os.path.join(ROOT, "Fus_bbbo_pipeline")
CODE = os.path.join(ROOT, "MRI 분석 코드")
NAME = "Light-eclipse"
EMAIL = "178866319+Light-eclipse@users.noreply.github.com"


def A(date, name):
    return os.path.join(ROOT, date, "2. 분석 결과", name)


def L(name):
    return os.path.join(CODE, name)


# (태그, 제목, 날짜, [(원본경로, 저장소경로)], 변경 요약 줄들)
VERSIONS = [
    ("v0.1", "DCE core pipeline", "2026-06-25", [
        (L("mri_dce.py"), "mri_dce.py"),
        (A("26-06-25", "_run_all_slices.py"), "run_dce.py"),
    ], [
        "Shared library (`mri_dce.py`): DICOM loading, series grouping into slabs,",
        "orientation classification, in-plane rigid registration with an edge-NCC guard,",
        "tissue masking, enhancement curves, `bbbo_metrics`, montages, and a Tk GUI.",
        "First per-date driver: resample every POST onto the PRE grid using IPP/IOP",
        "physical coordinates, build the slab by hand, register, and write every slice.",
    ]),
    ("v0.2", "Agilent scale correction, T2/T2*, EPT", "2026-08-20", [
        (A("26-08-20", "_run_0820.py"), "run_ept_t2.py"),
    ], [
        "**Agilent `FdfToDcm` renormalises pixel values per series.** The factor sits in",
        "private tag `(00E1,1001)`; true value = pixel / tag. Without this, PRE and POST",
        "are not on the same axis and every enhancement number is wrong. Verified against",
        "air-background noise, which is independent of Gd and receive gain — the background",
        "mask is computed on the *corrected* data (lowest 10%).",
        "Multi-echo relaxometry: T2 from `mems` (spin echo), T2* from `mgems`.",
        "Phase-based EPT: sigma = lap(phi) / (2 mu0 omega).",
        "Two constants are assumptions, not measurements, and are flagged in the docstring:",
        "echo spacing `DTE_MS` (no per-echo EchoTime in the headers) and the phase unit of",
        "`reconphs`. T2/T2* in ms scale linearly with the first; sigma scales inversely",
        "with the second.",
    ]),
    ("v0.3", "Mirror asymmetry detection", "2026-08-26", [
        (A("26-08-26", "_run_0826.py"), "run_dce.py"),
    ], [
        "Scale correction folded into the DCE driver.",
        "**Whole-tissue enhancement is not a BBBO measure.** Scalp and temporal muscle are",
        "most of the tissue voxels and muscle enhances 25-90% with Gd. Judging opening from",
        "that curve is meaningless.",
        "Mirror asymmetry instead: FUS-BBBO is usually unilateral while scalp and muscle are",
        "left-right symmetric, so subtracting the mirrored enhancement map leaves the",
        "unilateral component. The contralateral mirror ROI acts as an internal control, so",
        "the same definition as `bbbo_metrics` applies without manual ROIs.",
        "Also handles two data quirks: a TR-200 series on a different protocol (excluded from",
        "the DCE curve), and one animal's coronal series acquired with another animal's",
        "prescription (resampled back onto its own grid; correlation 0.852 -> 0.981).",
    ]),
    ("v0.4", "Tilted symmetry axis, cross-orientation check", "2026-09-11", [
        (A("26-09-11", "_run_0911.py"), "run_dce.py"),
        (A("26-09-11", "_compare_fse.py"), "compare_fse.py"),
    ], [
        "**Translation-only mirroring fails on a tilted head.** Fitted tilts ran -7.5 to",
        "+11.5 degrees; the residual misalignment leaked into the left-right difference as",
        "broad red/blue bands across the whole brain.",
        "The axis search now fits tilt and position together. An arbitrary line reflection is",
        "applied as a single affine, R = 2ddT - I, rather than rotate-flip-rotate (faster and",
        "less interpolation blur). Search runs on a 2x downsample, is restricted to the mask",
        "column centroid +-20 px, and rejects any candidate with under 75% mask overlap —",
        "without the constraint the axis wanders off. Results are cached per",
        "(animal, series, orientation) because the asymmetry is evaluated three times per slab.",
        "Candidates are then checked at the same physical coordinate in the orthogonal",
        "acquisition.",
        "**Every one of the 33 candidates this found was a registration residual** — an arc",
        "along the skull or brain surface, red on one side and blue on the other. Automatic",
        "metrics pass them. The figures have to be looked at.",
    ]),
    ("v0.5", "Symmetry-free local contrast, lesion depth", "2026-09-15", [
        (A("26-09-15", "_run_0915.py"), "run_dce.py"),
        (A("26-09-15", "_focus_depth.py"), "focus_depth.py"),
        (A("26-09-15", "_lesion_summary.py"), "lesion_summary.py"),
        (A("26-09-15", "_top4.py"), "top4.py"),
    ], [
        "Four targets per animal breaks the mirror method: with both hemispheres sonicated",
        "there is no contralateral control.",
        "`local_focus` needs no symmetry. Mask-weighted Gaussian background (sigma 20 px,",
        "about 2.3 mm), excess over 15 %p, control taken from a surrounding shell 4-12 px",
        "(0.5-1.4 mm) away.",
        "Depth below the brain surface: brain mask = tissue mask minus dark skull line,",
        "largest 3D component, per-slice hole filling, then a distance transform at the lesion",
        "centroid. `DARK_FRAC` had to go from 0.55 to 0.85 — at 0.55 the skull line is too thin",
        "and the mask swallows scalp and muscle (2694 mm3 against a correct 1919-2392 mm3).",
        "`lesion_summary.py` clusters candidates in patient coordinates and counts how many of",
        "the four independent acquisitions (GRE/FSE x axial/coronal) reproduce each one.",
        "Elongation and midline distance are **flags, not rejects** — using them as hard",
        "filters removed a visually confirmed real lesion.",
    ]),
    ("v0.6", "IVIM and T2 as independent contrasts", "2026-09-22", [
        (A("26-09-22", "_run_0922.py"), "run_dce.py"),
        (A("26-09-22", "_focus_depth.py"), "focus_depth.py"),
        (A("26-09-22", "_lesion_summary.py"), "lesion_summary.py"),
        (A("26-09-22", "_ivim_0922.py"), "ivim.py"),
        (A("26-09-22", "_t2_0922.py"), "t2_relaxometry.py"),
        (A("26-09-22", "_lesion_figs.py"), "lesion_figs.py"),
    ], [
        "IVIM (Agilent `epip`, SE-EPI): 11 series by b-value, 10-1000 s/mm2, acquired",
        "descending, with no separate b=0 series. Each slice has four images distinguished",
        "only by filename — image1 is the b~0 reference and images 2-4 are three diffusion",
        "directions. Direction vectors are absent from the headers, so the trace (geometric",
        "mean of the three) is computable but a tensor and FA are not.",
        "Normalising by the in-series b0 cancels both the scale tag and signal drift.",
        "Segmented fit: D from b >= 200, f = 1 - S_int, D* by log grid search, 300-sample",
        "bootstrap CI.",
        "**Voxelwise f and D* are not usable here** — noise is 3.4-3.6% of S0 against f ~ 0.038,",
        "and 52% of fitted D* land on the grid edge. ROI level only.",
        "MREPT NIfTI traps: `recon.nii` can be an md5-identical duplicate (no phase, so no EPT);",
        "the Agilent qform quaternion is invalid so `nib.load` raises `w2 should be positive`",
        "(read the header and data directly via `from_fileobj`); sform is zero so geometry has",
        "to be recovered by correlating with the GRE; no echo times anywhere; odd-even echo",
        "oscillation from stimulated echoes.",
    ]),
    ("v0.7", "Signal-decrease and haemorrhage review", "2026-09-22", [
        (L("negchange_scan.py"), "negchange/negchange_scan.py"),
        (L("negchange_diag.py"), "negchange/negchange_diag.py"),
        (L("negchange_shiftfix.py"), "negchange/negchange_shiftfix.py"),
        (L("negchange_localize.py"), "negchange/negchange_localize.py"),
        (L("spot_multimodal.py"), "negchange/spot_multimodal.py"),
        (L("hypo_scan.py"), "negchange/hypo_scan.py"),
    ], [
        "Every detector up to here thresholded on positive enhancement only, so a lesion that",
        "*darkens* was invisible to all of them.",
        "**Through-plane motion is the main cause of apparent signal loss.** `register_slab`",
        "with `mode=\"inplane\"` corrects in-plane only. Estimating the slice-direction shift by",
        "gradient-magnitude correlation over dz in [-1.5, +1.5] slices and correcting it removes",
        "the decrease entirely in one animal (30.5% -> 1.3% of brain voxels) and reduces it to a",
        "smooth front-to-back gradient in another.",
        "`hypo_scan.py` (local-contrast deficit) is kept but **should not be used on post-Gd",
        "data**: once the surface dura enhances, everything just inside it reads as a deficit,",
        "producing a false rim about 1 mm below the surface on every date.",
        "Focal dark spots are separated by their contrast across sequences. Haemorrhage is dark",
        "on a late T2 echo and blooms on EPI; a vein is dark on GRE, isointense on spin echo and",
        "enhances; CSF and sinus are dark on T1w, bright on T2 and fill in with Gd. All spots",
        "checked matched CSF, sinus or vein — none matched haemorrhage.",
        "Haemorrhage and BBBO are judged separately. Opening is judged by contrast",
        "extravasation; `haemorrhage seen, therefore opening confirmed` merges two outcomes.",
        "This GRE has TE 3.98 ms and is only weakly susceptibility-sensitive, so small",
        "microbleeds can be missed.",
    ]),
    ("v0.8", "Display rescaling and coil sensitivity correction", "2026-09-28", [
        (A("26-09-22", "_rescale.py"), "rescale.py"),
    ], [
        "Enhancement was hard to see on screen for two reasons.",
        "The grey window ran from air (14) to a level that clipped 11-22% of POST brain voxels,",
        "so the lower third of the ramp showed nothing and the top was blown out.",
        "**A receive-coil sensitivity gradient dominates the within-brain spread** — brain median",
        "falls from 379 to 213 across one slice, smoothly, a factor of 1.8. Narrowing the window",
        "alone just saturates the near-coil side first.",
        "Fix: fit a smooth multiplicative field to log(PRE) over brain voxels (3rd-order",
        "polynomial, normalised to 1 at the brain median) and **divide PRE and POST by the same",
        "field**. Dividing both by the same number leaves (POST-PRE)/PRE untouched; the script",
        "recomputes and prints the residual every run (order 1e-13 %p). The correction changes",
        "display only.",
        "Two brain-mask bugs surfaced and are fixed here. The erosion radius was in pixels, but",
        "GRE is 0.117 and FSE 0.156 mm/px, so the same number meant different strengths — it is",
        "now in mm. And the global threshold `PRE >= DARK_FRAC * tissue median` collapses on",
        "slices far from the coil, where the whole brain falls below it; one series lost slices",
        "0-3 down to 1680-2429 voxels against 17932 mid-slab. Mask and field are now re-fitted",
        "alternately, which restores those slices to 4989-11207.",
        "The window also has a noise floor: never narrower than 8x the PRE noise SD. Percentiles",
        "alone let the window collapse to the noise width on a low-contrast series (span 371",
        "against a noise SD of 73.6, so noise +-1 SD covered a quarter of the ramp).",
    ]),
]

GITIGNORE = """\
# Imaging data and results — never commit
*.dcm
*.nii
*.nii.gz
*.npz
*.fdf
*.png
*.jpg
*.pptx
*.hwp

# Per-run outputs
2. 분석 결과/
*_out/
brain_mask_check/
scale_test/
rescaled/
individual_rescaled/

# Python
__pycache__/
*.py[cod]
.venv/

# OS
.DS_Store
Thumbs.db
"""

LICENSE = """\
MIT License

Copyright (c) 2026 Light-eclipse

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

HEAD = """\
# Fus_bbbo_pipeline

Analysis pipeline for focused-ultrasound blood-brain barrier opening (FUS-BBBO) in the
rat, measured by dynamic contrast-enhanced MRI on a 9.4 T Agilent system.

Written for an ongoing preclinical study. The repository is a record of how the method
was actually built: each tag is the state of the pipeline on the date that stage was
first used on real data.

## Why the versions matter

Most of what is in here is not new code but a correction to something that looked
right and was not. The changelog states each one — what was measured, what it turned
out to be, and the number that settled it. Those notes are the point of the repository.

## Stages
"""

TAIL_TMPL = """

Full detail for each stage is in [CHANGELOG.md](CHANGELOG.md).

## Layout

| File | What it does |
|---|---|
| `mri_dce.py` | Shared library: DICOM loading, slab construction, registration, masks, enhancement curves, figures, GUI |
| `run_dce.py` | Per-date DCE driver. Replaced at each stage — its history is the history of the method |
| `run_ept_t2.py` | Multi-echo T2 / T2* and phase-based EPT |
| `focus_depth.py` | Brain mask and lesion depth below the brain surface |
| `lesion_summary.py` | Clusters candidates in patient coordinates, counts reproducing acquisitions |
| `top4.py` | Ranks candidates for a four-target design |
| `ivim.py` | IVIM: segmented D / f / D* fit from a b-value series |
| `t2_relaxometry.py` | T2 from MREPT multi-echo data |
| `lesion_figs.py` | Per-lesion figure across sequences and orientations |
| `rescale.py` | Display window, coil sensitivity correction, improved brain mask |
| `negchange/` | Signal-decrease review: through-plane shift, focal dark spots across contrasts |

## Requirements

Python 3, `numpy`, `scipy`, `pydicom`, `nibabel`, `matplotlib`, `SimpleITK`.
Korean text in the figures needs a Korean font (`_set_korean_font` picks one up on Windows).

## Known limitations

- **Paths are hardcoded.** Each driver carries the absolute path of the date it was
  written for. This is what was actually run, and it has been left that way rather than
  rewritten after the fact. Change the constants at the top of a driver to use it
  elsewhere.
- **The brain mask is not an anatomical segmentation.** It is a brightness threshold
  plus morphology, good enough to place a display window and to measure depth. Lateral
  and ventral boundaries still include some muscle.
- **Scanner axes, not brain axes.** The rat is prone, so scanner *axial* gives brain
  coronal sections and scanner *coronal* gives brain horizontal sections. Filenames use
  the scanner convention throughout.
- Slice numbers in all outputs are 0-indexed (DICOM slice N+1).

## Data

No imaging data is in this repository and none should be added. Raw DICOM, NIfTI,
intermediate `.npz` volumes and result figures stay local — about 25 GB at the time of
writing.
"""

CHANGELOG_HEAD = """\
# Changelog

Each version is the state of the pipeline on the date it was first used on real data.

**These commits are a reconstruction, not a recovered history.** They were assembled
from the per-date driver scripts that survive, which is why the sequence is faithful at
the stage level. Files that were edited in place rather than copied per date —
`mri_dce.py` above all — exist only in their final state and enter at v0.1 as they are
now.

"""


# 저장소 안에서의 이 스크립트 위치. --rebuild 때 자기 자신을 되살리는 데 쓴다.
SELF = os.path.join("tools", "build_history.py")

TOOLING = """

## Tooling

`tools/build_history.py` holds the version table and rebuilt this history from the
per-date scripts. Adding a stage means adding one entry to `VERSIONS` and running it;
it commits only what has no tag yet. See its docstring.

Tags mark pipeline stages. Commits without a tag are tooling or documentation.
"""


def sh(*args, **kw):
    r = subprocess.run(args, cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", **kw)
    if r.returncode != 0:
        print("!! ", " ".join(args))
        print(r.stdout, r.stderr)
        sys.exit(1)
    return r.stdout


def existing_tags():
    r = subprocess.run(["git", "tag"], cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return set(r.stdout.split())


def write_docs(upto, tooling):
    """README 와 CHANGELOG 를 VERSIONS[:upto] 기준으로 다시 쓴다."""
    rows, chunks = [], []
    for tag, title, date, files, notes in VERSIONS[:upto]:
        rows.append(f"| [`{tag}`](../../releases/tag/{tag}) | {date} | {title} |")
        chunks.append(f"## {tag} — {title}\n\n*{date}*\n\n"
                      + "\n".join(notes) + "\n\nFiles: "
                      + ", ".join(f"`{d}`" for _, d in files) + "\n")
    table = ("\n| Tag | First used | What it added |\n|---|---|---|\n"
             + "\n".join(rows) + "\n")
    body = HEAD + table + TAIL_TMPL + (TOOLING if tooling else "")
    with open(os.path.join(REPO, "README.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    with open(os.path.join(REPO, "CHANGELOG.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(CHANGELOG_HEAD + "\n".join(reversed(chunks)))


def commit(msg, date):
    sh("git", "add", "-A")
    env = dict(os.environ, GIT_AUTHOR_DATE=f"{date}T12:00:00+09:00",
               GIT_COMMITTER_DATE=f"{date}T12:00:00+09:00",
               GIT_AUTHOR_NAME=NAME, GIT_AUTHOR_EMAIL=EMAIL,
               GIT_COMMITTER_NAME=NAME, GIT_COMMITTER_EMAIL=EMAIL)
    subprocess.run(["git", "commit", "-q", "-F", "-"], cwd=REPO, env=env,
                   input=msg, text=True, encoding="utf-8")
    return env


def apply_version(i, tooling):
    tag, title, date, files, notes = VERSIONS[i]
    for src, dst in files:
        if not os.path.exists(src):
            print("원본 없음:", src)
            sys.exit(1)
        dp = os.path.join(REPO, dst)
        os.makedirs(os.path.dirname(dp) or REPO, exist_ok=True)
        shutil.copy2(src, dp)
    write_docs(i + 1, tooling)
    body = "\n".join("- " + n.replace("**", "") for n in notes[:4])
    env = commit(f"{tag}  {title}\n\n{body}\n", date)
    subprocess.run(["git", "tag", "-a", tag, "-m", f"{tag} {title}"],
                   cwd=REPO, env=env, capture_output=True)
    print(f"  {tag}  {title}")


def clear_repo():
    """폴더 자체는 탐색기 등이 잡고 있을 수 있으므로 안을 비우기만 한다."""
    def _force(fn, path, _):
        os.chmod(path, 0o700)
        fn(path)

    os.makedirs(REPO, exist_ok=True)
    for e in os.listdir(REPO):
        q = os.path.join(REPO, e)
        if os.path.isdir(q):
            shutil.rmtree(q, onerror=_force)
        else:
            os.chmod(q, 0o700)
            os.remove(q)


def main():
    argv = sys.argv[1:]
    rebuild = "--rebuild" in argv

    if rebuild:
        if "--force" not in argv:
            print("--rebuild 는 모든 커밋 SHA 를 바꾼다. 이미 푸시했다면 강제 푸시가\n"
                  "필요하다. 그래도 하려면 --force 를 같이 준다.")
            sys.exit(1)
        me = open(os.path.abspath(__file__), encoding="utf-8").read()
        clear_repo()
        subprocess.run(["git", "init", "-b", "main"], cwd=REPO,
                       capture_output=True, text=True)
        for name, text in ((".gitignore", GITIGNORE), ("LICENSE", LICENSE)):
            with open(os.path.join(REPO, name), "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
        for i in range(len(VERSIONS)):
            apply_version(i, tooling=False)
        # 자기 자신을 마지막에 되살린다 (태그 없는 도구 커밋)
        os.makedirs(os.path.join(REPO, "tools"), exist_ok=True)
        with open(os.path.join(REPO, SELF), "w", encoding="utf-8", newline="\n") as f:
            f.write(me)
        write_docs(len(VERSIONS), tooling=True)
        commit("tooling: history build script\n\n"
               "- The version table and the script that assembled these commits.\n",
               VERSIONS[-1][2])
        print("  tooling: history build script")
    else:
        have = existing_tags()
        todo = [i for i, v in enumerate(VERSIONS) if v[0] not in have]
        if not todo:
            print("새로 올릴 버전이 없다. VERSIONS 에 항목을 추가한 뒤 다시 실행한다.")
            return
        for i in todo:
            apply_version(i, tooling=os.path.exists(os.path.join(REPO, SELF)))

    print("\n" + sh("git", "log", "--oneline", "--decorate"))


if __name__ == "__main__":
    main()
