# Round 1 submission checklist

Deadline: **31 October 2026**. Registration closes **15 October 2026**.

The Drive folder must be shared as "Anyone with the link can view". Loose links are not reviewed:
every link goes inside one master PDF.

## Drive folder contents

| File | What goes in it | Source |
|---|---|---|
| `01_Technical_Abstract.pdf` (max 5 pages) | Architecture, pipeline, SOP, approach, results, credits | Fill in `docs/TECHNICAL_ABSTRACT.md` and export to PDF |
| `02_Links_and_Proof_of_Concept.pdf` | Web viewer URL, repository URL, video URL, each with one line saying what it is | Write once the three exist |
| `03_Demo_Video.mp4` (3 to 5 minutes) | Scope, capture, walkthrough | Shot list below |
| `04_Code/` or a repository link | This repository, with setup instructions and `tests/sample/` | Push to GitHub; make it public |

## Hosting the viewer

```bash
cd viewer && npm install && npm run build
```

`viewer/dist` is the whole site. Any static host works:

- **GitHub Pages**: push `viewer/dist` to a `gh-pages` branch, or build it in an Action.
- **Netlify or Cloudflare Pages**: drag the `dist` folder onto the dashboard.

Open the link on a laptop and on a phone before submitting. Then open it in a private window to
confirm it needs no login.

## Before you submit

- [ ] Real capture of your own space run through the whole pipeline; the sample scene replaced.
- [ ] Report card numbers from that run copied into the abstract (section 6).
- [ ] Frame rate read off the report card on a laptop and on a phone (`?bench=1` replays the
      capture path and reports average and worst-1% frame rate).
- [ ] Every third-party tool credited with its licence in the abstract (competition rule 4).
- [ ] MapAnything run with `--apache` weights (the default weights are non-commercial).
- [ ] Repository README states what was and was not tested, updated for your real run.
- [ ] Drive folder permission is "Anyone with the link can view".

## Video shot list (aim for 4 minutes)

| Time | Shot | Say |
|---|---|---|
| 0:00 to 0:25 | Title, then the finished twin being walked in the browser | What it is: phone video to a walkable, metric twin, with a loop that checks the capture. |
| 0:25 to 1:05 | Phone screen and over-the-shoulder footage of the capture | The SOP: lock exposure, marker on the floor, pass A looking in, pass B looking out, doorways. |
| 1:05 to 1:35 | Lint report on screen | Seconds after filming: here it caught where I rushed. Keyframes chosen by parallax. |
| 1:35 to 2:20 | Recapture ticket, then you filming item 1, then the second ticket | The draft found this corner weak, told me where to stand, 20 seconds of footage fixed it. Show the before and after numbers. |
| 2:20 to 2:45 | Pipeline diagram | Metric frame from one marker; draft poses guide COLMAP; trained in the final frame. |
| 2:45 to 3:45 | Viewer: WASD walk, bump into a wall, click the plan to auto-walk, measure a doorway against a real tape measure, show it on a phone | Collision and the plan come from one grid. Distances are real. |
| 3:45 to 4:00 | Report card close-up | Every number, measured. What comes next: POIs and navigation on the same grid. |

The tape-measure shot is worth the effort: it is the one thing a viewer cannot fake.
