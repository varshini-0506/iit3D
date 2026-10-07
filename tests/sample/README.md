# Sample data

Everything here comes from a **synthetic** room (`nammatwin/synth.py`): a 5 x 4 m box with a table,
a partitioned alcove and a printed-size ArUco marker on the floor, rendered as a phone-style
walkthrough. It is not a real place. It exists because it has exact ground truth, so the pipeline
can be checked rather than eyeballed.

| File | What it is |
|---|---|
| `clips/walk.mp4` | 79 s two-pass walkthrough. Contains a whip-pan at 30 s and motion blur at 60 s on purpose. The walker never enters the alcove. |
| `clips/recapture_01.mp4` | 25 s top-up clip that steps into the alcove, as the first ticket asks. |
| `expected/lint_timeline_walk.png` | The linter's timeline for `walk.mp4`, with both defects flagged. |
| `expected/ticket_before_recapture.png` | Coverage map and recapture ticket after the first clip: verdict RECAPTURE, arrows into the alcove. |
| `expected/ticket_after_recapture.png` | The same after the top-up clip: verdict READY. |
| `expected/coverage_*.json` | The numbers behind both tickets. |
| `expected/transform.json` | The alignment found, including the 3% scale error the marker corrected. |
| `expected/floorplan.png` | Floor plan derived from the splat. |
| `expected/report.json` | The report card shown in the viewer. |

Regenerate everything, including the viewer's sample scene:

```bash
python -m nammatwin sample --out work_sample --scene viewer/public/scene
```

Try the linter alone on the clips:

```bash
python -m nammatwin lint tests/sample/clips/walk.mp4 --out work_lint
```

The draft reconstruction in the sample is ground truth standing in for MapAnything, and the splat is
written directly instead of trained, because both of those stages need a GPU. `tests/test_pipeline_script.sh`
runs the real pipeline script with the same stand-ins, including real COLMAP bundle adjustment on
the rendered frames.
