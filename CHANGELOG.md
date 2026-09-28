# Changelog

Each version is the state of the pipeline on the date it was first used on real data.

**These commits are a reconstruction, not a recovered history.** They were assembled
from the per-date driver scripts that survive, which is why the sequence is faithful at
the stage level. Files that were edited in place rather than copied per date —
`mri_dce.py` above all — exist only in their final state and enter at v0.1 as they are
now.

## v0.8 — Display rescaling and coil sensitivity correction

*2026-09-28*

Enhancement was hard to see on screen for two reasons.
The grey window ran from air (14) to a level that clipped 11-22% of POST brain voxels,
so the lower third of the ramp showed nothing and the top was blown out.
**A receive-coil sensitivity gradient dominates the within-brain spread** — brain median
falls from 379 to 213 across one slice, smoothly, a factor of 1.8. Narrowing the window
alone just saturates the near-coil side first.
Fix: fit a smooth multiplicative field to log(PRE) over brain voxels (3rd-order
polynomial, normalised to 1 at the brain median) and **divide PRE and POST by the same
field**. Dividing both by the same number leaves (POST-PRE)/PRE untouched; the script
recomputes and prints the residual every run (order 1e-13 %p). The correction changes
display only.
Two brain-mask bugs surfaced and are fixed here. The erosion radius was in pixels, but
GRE is 0.117 and FSE 0.156 mm/px, so the same number meant different strengths — it is
now in mm. And the global threshold `PRE >= DARK_FRAC * tissue median` collapses on
slices far from the coil, where the whole brain falls below it; one series lost slices
0-3 down to 1680-2429 voxels against 17932 mid-slab. Mask and field are now re-fitted
alternately, which restores those slices to 4989-11207.
The window also has a noise floor: never narrower than 8x the PRE noise SD. Percentiles
alone let the window collapse to the noise width on a low-contrast series (span 371
against a noise SD of 73.6, so noise +-1 SD covered a quarter of the ramp).

Files: `rescale.py`

## v0.7 — Signal-decrease and haemorrhage review

*2026-09-22*

Every detector up to here thresholded on positive enhancement only, so a lesion that
*darkens* was invisible to all of them.
**Through-plane motion is the main cause of apparent signal loss.** `register_slab`
with `mode="inplane"` corrects in-plane only. Estimating the slice-direction shift by
gradient-magnitude correlation over dz in [-1.5, +1.5] slices and correcting it removes
the decrease entirely in one animal (30.5% -> 1.3% of brain voxels) and reduces it to a
smooth front-to-back gradient in another.
`hypo_scan.py` (local-contrast deficit) is kept but **should not be used on post-Gd
data**: once the surface dura enhances, everything just inside it reads as a deficit,
producing a false rim about 1 mm below the surface on every date.
Focal dark spots are separated by their contrast across sequences. Haemorrhage is dark
on a late T2 echo and blooms on EPI; a vein is dark on GRE, isointense on spin echo and
enhances; CSF and sinus are dark on T1w, bright on T2 and fill in with Gd. All spots
checked matched CSF, sinus or vein — none matched haemorrhage.
Haemorrhage and BBBO are judged separately. Opening is judged by contrast
extravasation; `haemorrhage seen, therefore opening confirmed` merges two outcomes.
This GRE has TE 3.98 ms and is only weakly susceptibility-sensitive, so small
microbleeds can be missed.

Files: `negchange/negchange_scan.py`, `negchange/negchange_diag.py`, `negchange/negchange_shiftfix.py`, `negchange/negchange_localize.py`, `negchange/spot_multimodal.py`, `negchange/hypo_scan.py`

## v0.6 — IVIM and T2 as independent contrasts

*2026-09-22*

IVIM (Agilent `epip`, SE-EPI): 11 series by b-value, 10-1000 s/mm2, acquired
descending, with no separate b=0 series. Each slice has four images distinguished
only by filename — image1 is the b~0 reference and images 2-4 are three diffusion
directions. Direction vectors are absent from the headers, so the trace (geometric
mean of the three) is computable but a tensor and FA are not.
Normalising by the in-series b0 cancels both the scale tag and signal drift.
Segmented fit: D from b >= 200, f = 1 - S_int, D* by log grid search, 300-sample
bootstrap CI.
**Voxelwise f and D* are not usable here** — noise is 3.4-3.6% of S0 against f ~ 0.038,
and 52% of fitted D* land on the grid edge. ROI level only.
MREPT NIfTI traps: `recon.nii` can be an md5-identical duplicate (no phase, so no EPT);
the Agilent qform quaternion is invalid so `nib.load` raises `w2 should be positive`
(read the header and data directly via `from_fileobj`); sform is zero so geometry has
to be recovered by correlating with the GRE; no echo times anywhere; odd-even echo
oscillation from stimulated echoes.

Files: `run_dce.py`, `focus_depth.py`, `lesion_summary.py`, `ivim.py`, `t2_relaxometry.py`, `lesion_figs.py`

## v0.5 — Symmetry-free local contrast, lesion depth

*2026-09-15*

Four targets per animal breaks the mirror method: with both hemispheres sonicated
there is no contralateral control.
`local_focus` needs no symmetry. Mask-weighted Gaussian background (sigma 20 px,
about 2.3 mm), excess over 15 %p, control taken from a surrounding shell 4-12 px
(0.5-1.4 mm) away.
Depth below the brain surface: brain mask = tissue mask minus dark skull line,
largest 3D component, per-slice hole filling, then a distance transform at the lesion
centroid. `DARK_FRAC` had to go from 0.55 to 0.85 — at 0.55 the skull line is too thin
and the mask swallows scalp and muscle (2694 mm3 against a correct 1919-2392 mm3).
`lesion_summary.py` clusters candidates in patient coordinates and counts how many of
the four independent acquisitions (GRE/FSE x axial/coronal) reproduce each one.
Elongation and midline distance are **flags, not rejects** — using them as hard
filters removed a visually confirmed real lesion.

Files: `run_dce.py`, `focus_depth.py`, `lesion_summary.py`, `top4.py`

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
