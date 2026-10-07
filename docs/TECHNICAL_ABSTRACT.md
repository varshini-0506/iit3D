# NammaTwin: a closed-loop pipeline from phone video to a walkable, metric 3D twin

*Technical abstract draft for Round 1. Replace every `[FILL]` with numbers from your own capture
(they are all on the viewer's report card and in `work/`), add your team name and ID, and export to
PDF. Keep it to five pages: the figures listed at the end fit if each is half a column wide.*

**Team:** `[FILL]` &nbsp; **Team ID:** `[FILL]` &nbsp; **Viewer:** `[FILL url]` &nbsp; **Code:** `[FILL url]`

## 1. Problem and approach

A Gaussian-splat twin is only as good as its footage, and a non-expert cannot tell good footage from
bad until training has finished. Most pipelines are therefore open loop: film, wait, and discover
the holes. We close the loop. Seconds after filming, a linter grades the video. Minutes later, a
draft reconstruction is audited for coverage and the system issues a *recapture ticket*: a plan of
the space with numbered spots, each saying where to stand, which way to face and how to move. Only
when the audit passes do we spend GPU time on the final model. The twin is built in a metric,
gravity-aligned frame from the start, so walking speed, eye height, the floor plan and measured
distances in the viewer are real, and Rounds 2 and 3 inherit a ready navigation map.

## 2. System architecture

Seven stages; stages 1 to 4 form the loop.

1. **Capture linter** (CPU, seconds). Per frame: an edge-sharpness ratio that is independent of how
   much texture is in view, median optical-flow speed in image-widths per second, the share of the
   frame with trackable corners, and exposure clipping. Keyframes are taken per unit of *parallax*
   (7% of image width), choosing the sharpest frame in each window, so standing still adds nothing
   and blurred frames are skipped.
2. **Draft reconstruction** (GPU, minutes). MapAnything predicts metric camera poses and dense
   points for all keyframes in one feed-forward pass.
3. **Canonical frame** (CPU). RANSAC finds the floor (the lowest large near-horizontal plane with
   the cameras above it); histogram-entropy minimisation aligns walls to the axes; a printed ArUco
   marker, triangulated from the draft poses, sets or checks the scale. Output: one similarity
   transform to "Z up, floor at 0, walls along X/Y, metres, origin at the start".
4. **Coverage audit** (CPU). For each surface sample we count the keyframes that see it (with a
   z-buffer for occlusion) and compute an effective baseline angle, 2·acos(|Σd̂|/n), from the sum of
   unit view directions. Samples with fewer than 4 views or under 12° are *weak*. Separately,
   observed floor that borders never-observed space with no wall between is an open *frontier*
   (frontier detection from robot exploration, used here to direct a person). Weak clusters and
   frontiers become ticket items with a standing position chosen on the walkable grid with a clear
   line of sight.
5. **Final poses** (CPU). COLMAP bundle adjustment on full-resolution keyframes, matching only the
   image pairs that the draft says stood near each other and looked the same way. The result is
   moved onto the draft's frame by a robust similarity fit of camera centres.
6. **Training** (GPU). gsplat with the MCMC strategy and a fixed Gaussian budget, trained directly
   in the canonical frame with every eighth keyframe held out for evaluation.
7. **Packaging** (CPU). Debris outside the room box is removed, the splat is compressed to SOG, and
   a 5 cm occupancy grid is derived from the trained Gaussians: floor band, body-height obstacle
   band, and the capture path carved as known-free space.

**Web viewer.** A static site (three.js with the Spark splat renderer): WASD and mouse-look with
real eye height, collision and wall sliding on the grid, a live floor plan, click-to-walk by A*
shortest path with string pulling, a measuring tool, a replay of the capture path, touch controls,
adaptive render resolution, and the report card. No install and no server-side code.

## 3. Capture SOP

One printed marker on the floor; exposure and focus locked; landscape, main lens. Two passes per
room at one slow step per second: **A**, round the walls looking in; **B**, round the middle looking
out, stepping into alcoves. Walk through doorways facing forward, then look back. Finish where you
started. About one minute per 10 m². The full procedure, with the reasons and the pass marks the
linter and audit apply, is in `docs/SOP.md`.

## 4. What is new

- **Recapture ticket from a draft.** Coverage is audited before training, with view count and
  parallax per surface sample plus frontier detection, and turned into instructions a non-expert
  can follow.
- **Texture-independent blur measure and parallax-based keyframes**, so a blank wall is reported as
  texture-poor rather than blurry, and frame selection follows motion rather than the clock.
- **Draft-guided matching for bundle adjustment**, which makes COLMAP's cost linear in keyframes
  and links loop closures and recapture clips without a vocabulary tree.
- **One metric grid for collision, floor plan and routing**, derived from the splat itself.
- **A report card** that publishes capture, coverage, geometry, fidelity and device performance
  beside the scene.

## 5. Verification on ground truth

Before any real capture, every CPU stage was tested on a synthetic room with exact ground truth
(19 automated tests in the repository):

| Test | Result |
|---|---|
| Injected whip-pan and motion blur in a 79 s clip | both flagged; linting took about 8 s on a laptop CPU |
| Model handed over tilted, rotated and rescaled (3 random cases) | floor flat to under 2 cm; room size within 10 cm |
| Draft 3.00% out of scale | marker measured 2.95 to 3.04% |
| Alcove never entered by the walker | ticket pointed into it; verdict RECAPTURE |
| 25 s top-up clip as instructed | walls and furniture 96.7% to 99.4% well observed; verdict READY |
| COLMAP with draft-guided pairs, 343 keyframes | 337 registered using 9% of exhaustive pairs; 0.6 cm RMS from the draft |
| SOG compression | 7.6 MB to 0.66 MB, frame preserved |

## 6. Results on our space

`[FILL: one sentence describing the space: rooms, floor area, lighting.]`

| Measure | Value |
|---|---|
| Footage, clips, keyframes | `[FILL]` |
| Capture score; share blurred or too fast | `[FILL]` |
| Recapture rounds; items on first and last ticket | `[FILL]` |
| Walls, furniture and floor well observed, before and after top-up | `[FILL]` |
| Scale source; draft model's own scale error | `[FILL]` |
| A real distance measured by tape vs in the viewer | `[FILL]` |
| Keyframes registered by COLMAP; draft vs final pose RMS | `[FILL]` |
| Held-out PSNR / SSIM / LPIPS | `[FILL]` |
| Gaussians; download size | `[FILL]` |
| Load time and frame rate, laptop and phone (`?bench=1`) | `[FILL]` |
| Time per stage; total from end of filming to live link | `[FILL]` |

`[FILL: two or three sentences on what went wrong and what the loop caught.]`

## 7. Limitations and next rounds

Mirrors, glass and blank walls remain hard for image-based reconstruction; the linter warns about
texture-poor stretches but cannot fix them. The audit sees only surfaces present in the draft, which
is why frontier detection is needed for never-seen areas. Thresholds were set on synthetic footage
and should be re-tuned on real captures `[FILL: say whether you did]`.

Round 2 builds on what exists: POIs are points in the metric frame, and the viewer already routes
on the grid. Round 3: per-zone clips linted on site, draft and audit run before leaving the venue.

## 8. Credits and licences

MapAnything (Apache-2.0 code and `map-anything-apache` weights), COLMAP and pycolmap (BSD-3),
gsplat (Apache-2.0), splat-transform (MIT), Spark and three.js (MIT), OpenCV (Apache-2.0), NumPy,
SciPy and Matplotlib (BSD-style), Barlow typeface (SIL OFL 1.1). All other code is our own, MIT
licensed.

## Figures to include

1. Pipeline diagram (README, "Pipeline").
2. Lint timeline (`work/lint_timeline_c00.png`).
3. Recapture ticket before and after (`work/audit/history/pass_1` and the last pass).
4. Viewer screenshot with plan and report card.
5. A photo of the space beside the same view in the viewer.
