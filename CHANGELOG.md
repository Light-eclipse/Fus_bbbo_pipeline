# Changelog

Each version is the state of the pipeline on the date it was first used on real data.

**These commits are a reconstruction, not a recovered history.** They were assembled
from the per-date driver scripts that survive, which is why the sequence is faithful at
the stage level. Files that were edited in place rather than copied per date —
`mri_dce.py` above all — exist only in their final state and enter at v0.1 as they are
now.

## v0.4 — Tilted symmetry axis, cross-orientation check

*2026-09-11*

**Translation-only mirroring fails on a tilted head.** Fitted tilts ran -7.5 to
+11.5 degrees; the residual misalignment leaked into the left-right difference as
broad red/blue bands across the whole brain.
The axis search now fits tilt and position together. An arbitrary line reflection is
applied as a single affine, R = 2ddT - I, rather than rotate-flip-rotate (faster and
less interpolation blur). Search runs on a 2x downsample, is restricted to the mask
column centroid +-20 px, and rejects any candidate with under 75% mask overlap —
without the constraint the axis wanders off. Results are cached per
(animal, series, orientation) because the asymmetry is evaluated three times per slab.
Candidates are then checked at the same physical coordinate in the orthogonal
acquisition.
**Every one of the 33 candidates this found was a registration residual** — an arc
along the skull or brain surface, red on one side and blue on the other. Automatic
metrics pass them. The figures have to be looked at.

Files: `run_dce.py`, `compare_fse.py`

## v0.3 — Mirror asymmetry detection

*2026-08-26*

Scale correction folded into the DCE driver.
**Whole-tissue enhancement is not a BBBO measure.** Scalp and temporal muscle are
most of the tissue voxels and muscle enhances 25-90% with Gd. Judging opening from
that curve is meaningless.
Mirror asymmetry instead: FUS-BBBO is usually unilateral while scalp and muscle are
left-right symmetric, so subtracting the mirrored enhancement map leaves the
unilateral component. The contralateral mirror ROI acts as an internal control, so
the same definition as `bbbo_metrics` applies without manual ROIs.
Also handles two data quirks: a TR-200 series on a different protocol (excluded from
the DCE curve), and one animal's coronal series acquired with another animal's
prescription (resampled back onto its own grid; correlation 0.852 -> 0.981).

Files: `run_dce.py`

## v0.2 — Agilent scale correction, T2/T2*, EPT

*2026-08-20*

**Agilent `FdfToDcm` renormalises pixel values per series.** The factor sits in
private tag `(00E1,1001)`; true value = pixel / tag. Without this, PRE and POST
are not on the same axis and every enhancement number is wrong. Verified against
air-background noise, which is independent of Gd and receive gain — the background
mask is computed on the *corrected* data (lowest 10%).
Multi-echo relaxometry: T2 from `mems` (spin echo), T2* from `mgems`.
Phase-based EPT: sigma = lap(phi) / (2 mu0 omega).
Two constants are assumptions, not measurements, and are flagged in the docstring:
echo spacing `DTE_MS` (no per-echo EchoTime in the headers) and the phase unit of
`reconphs`. T2/T2* in ms scale linearly with the first; sigma scales inversely
with the second.

Files: `run_ept_t2.py`

## v0.1 — DCE core pipeline

*2026-06-25*

Shared library (`mri_dce.py`): DICOM loading, series grouping into slabs,
orientation classification, in-plane rigid registration with an edge-NCC guard,
tissue masking, enhancement curves, `bbbo_metrics`, montages, and a Tk GUI.
First per-date driver: resample every POST onto the PRE grid using IPP/IOP
physical coordinates, build the slab by hand, register, and write every slice.

Files: `mri_dce.py`, `run_dce.py`
