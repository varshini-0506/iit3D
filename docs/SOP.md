# Capture SOP

How to film an indoor space so that it reconstructs well, and how the pipeline checks that you did.

The procedure has one idea behind it: **a surface reconstructs well only if the camera saw it from
several positions that are spread apart.** Standing in the middle of a room and spinning gives every
wall many views but no parallax, and the twin dissolves when you walk through it. The two-pass walk
below gives every surface views from near and far and from both sides. The linter and the coverage
audit then check the footage instead of trusting it.

## 1. Kit

| Item | Setting | Why |
|---|---|---|
| Any recent phone | Main (1x) lens, **landscape**, 4K 30 fps or 1080p 30 fps | The ultrawide lens distorts heavily; landscape sees more wall per frame. |
| Exposure and focus | **Locked** (long-press on a mid-bright wall until AE/AF lock shows) | Auto-exposure pumping changes surface colours between frames. |
| HDR / Dolby Vision video | Off | Tone mapping varies frame to frame. |
| Stabilisation | Standard is fine; turn off "action" or "super steady" modes | Those crop and warp the image. |
| Lens | Wiped clean | Smears read as blur everywhere. |
| One printed ArUco marker | `python -m nammatwin marker`, printed on A4, **edge of the black square measured with a ruler** | Gives metric scale without a tape measure. Bigger is more accurate. |

## 2. Prepare the space (2 minutes)

1. Switch on every light. Half-close blinds on windows that would blow out.
2. Open the doors of rooms you are including. Close the doors of rooms you are leaving out.
3. Stop anything that moves: ceiling fans, curtains in a draught, a TV, pets, people.
4. Tape the marker flat on the floor in open space, away from furniture, where you will pass it
   early in the walk.
5. Mirrors and glass reconstruct badly. You cannot fix that at capture; avoid lingering on them.

## 3. The walk

Hold the phone in both hands at chest height. Walk at about one slow step per second. **Turn with
your feet, not your wrists**: a full turn should take 15 to 20 seconds.

**First 20 seconds: the marker.** Start about 1 m from the marker, tilted down at it, and take
three or four slow steps in an arc around it. Carry straight on into pass A without stopping the
recording.

**Pass A: round the walls, looking in.** Walk the perimeter of the room about 60 cm off the walls,
with the camera pointed across the room at the opposite side. Sweep gently left and right, and let
the tilt drift from the floor in front of you to the top of the far wall.

**Pass B: round the middle, looking out.** Walk a smaller loop round the centre of the room (round
the table, the bed, the sofa) with the camera pointed outward at the nearest wall. Step into every
alcove and behind every large piece of furniture. This is the pass that sees what pass A hid.

**Doorways.** Walk through slowly, facing forward. One step in, turn and look back at the doorway
you came through. That view is what ties the two rooms together. Then do passes A and B in the new
room.

**Finish where you started,** looking the way you first looked.

One continuous clip for the whole home is best. If you must stop, start the next clip standing
somewhere the previous clip already filmed, looking at something it already saw.

### Do and do not

| Do | Do not |
|---|---|
| Keep an edge, corner or piece of furniture in every frame | Fill the frame with a blank wall, a window or the floor alone |
| Stay at least 1 m from what you are filming | Film close-ups; they blur and have nothing to match |
| Move sideways past things | Stand still and pivot |
| Tilt down to the floor and up to the ceiling as you go | Hold one fixed tilt for the whole walk |
| Walk the same route again if you bumped or rushed | Whip the phone round to "quickly get" something |

### Time budget

About 1 minute per 10 m² of floor, plus 15 seconds per doorway. A two-bedroom flat is 6 to 8
minutes of footage.

## 4. Check, top up, finish

| Step | Command | You get | Pass mark |
|---|---|---|---|
| Lint | `run_pipeline.sh video.mp4 work` (first stage) | `work/lint_report.html`: a timeline of sharpness, speed and trackable detail, with each problem stretch and what to do | Verdict `PASS` or `WARN`. On `FAIL`, reshoot. |
| Audit | same command, last stage | `work/audit/recapture_ticket.html`: a plan of the space, green where surfaces are well observed and red where weak, with numbered stand-here arrows | Verdict `READY` |
| Top up | `run_pipeline.sh --append clip.mp4 work` | The ticket again, with the new footage counted | Repeat until `READY`; usually one round |
| Finish | `run_pipeline.sh --finish work` | The trained, compressed twin and its report card in the viewer | See the report card |

What the checks measure, with their default thresholds (all adjustable):

- **Too fast**: the image moves more than 0.5 image-widths per second (about 30° per second of
  turning). Above 1.0 is graded `fail`.
- **Blurry**: fine detail at edges drops below 45% of what neighbouring frames show.
- **Texture-poor**: trackable corners cover less than 35% of the frame.
- **Keyframes** are taken every 7% of an image-width of parallax, choosing the sharpest frame in
  each window. Standing still adds none.
- **Weak surface**: seen in fewer than 4 keyframes, or only from directions less than 12° apart.
- **Open frontier**: observed floor that borders never-observed space with no wall between.
- **Ready**: at least 90% of floor, walls and furniture well observed, at least 90% of the floor
  boundary closed, and no weak patch of 0.25 m² or more. The ceiling is reported, and a
  tilted-up lap is suggested if it is under-covered, but it does not block training.

## 5. Scale

With the marker, scale comes from its measured edge whenever it was seen in five or more keyframes
with consistent edges; the report card then also says how far the draft model's own scale was off.
Without a marker the draft model's metric scale is used as is; `CAMERA_HEIGHT=1.45` is a cruder
fallback that assumes you held the phone at that height.

Expect about 1% from an A4-size marker filmed from 1 m. Accuracy falls with small or distant
markers.

## 6. On site (Round 3)

- Shoot one clip per zone, each starting inside the previous zone. Lint each clip as soon as it is
  shot; it takes seconds on a laptop.
- Run the draft and the audit while still in the building, so that the recapture ticket can be
  acted on before you leave.
- Put the marker in the first zone and film it well. One marker sets the scale for the whole model
  (the pipeline uses the marker ID it saw most often).
- Long corridors with blank walls are the hardest case. Walk them twice, once down each side,
  angled at the opposite wall, and keep doors, signs and floor patterns in frame.
