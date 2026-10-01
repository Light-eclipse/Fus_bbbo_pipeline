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

| Tag | First used | What it added |
|---|---|---|
| [`v0.1`](../../releases/tag/v0.1) | 2026-06-25 | DCE core pipeline |
| [`v0.2`](../../releases/tag/v0.2) | 2026-08-20 | Agilent scale correction, T2/T2*, EPT |
| [`v0.3`](../../releases/tag/v0.3) | 2026-08-26 | Mirror asymmetry detection |
| [`v0.4`](../../releases/tag/v0.4) | 2026-09-11 | Tilted symmetry axis, cross-orientation check |
| [`v0.5`](../../releases/tag/v0.5) | 2026-09-15 | Symmetry-free local contrast, lesion depth |
| [`v0.6`](../../releases/tag/v0.6) | 2026-09-22 | IVIM and T2 as independent contrasts |
| [`v0.7`](../../releases/tag/v0.7) | 2026-09-22 | Signal-decrease and haemorrhage review |
| [`v0.8`](../../releases/tag/v0.8) | 2026-09-28 | Display rescaling and coil sensitivity correction |
| [`v0.9`](../../releases/tag/v0.9) | 2026-10-01 | Contrast delivery check, EPT phase tests, DICOM T2/T2* |


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
| `t2_relaxometry.py` | T2 from MREPT multi-echo NIfTI (kept for comparison with `t2_dicom.py`) |
| `lesion_figs.py` | Per-lesion figure across sequences and orientations |
| `rescale.py` | Display window, coil sensitivity correction, improved brain mask |
| `verify.py` | Pre-analysis check: protocol, scale tag, acquisition time and geometry per series |
| `t2_dicom.py` | T2 / T2* fitted directly from DICOM multi-echo series |
| `ept/` | Phase-file validity tests for EPT: noise ratio, split-half, echo decomposition, sigma uncertainty |
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


## Tooling

`tools/build_history.py` holds the version table and rebuilt this history from the
per-date scripts. Adding a stage means adding one entry to `VERSIONS` and running it;
it commits only what has no tag yet. See its docstring.

Tags mark pipeline stages. Commits without a tag are tooling or documentation.
