// The twin report card: what the pipeline measured, plus what this device is doing right now.

const fmt = (v, digits = 0) => (v == null || Number.isNaN(v) ? null : Number(v).toFixed(digits));

function duration(s) {
  if (s == null) return null;
  const m = Math.floor(s / 60), r = Math.round(s % 60);
  return m ? `${m} min ${r} s` : `${r} s`;
}

function tone(value, good, fair) {
  return value >= good ? "ok" : value >= fair ? "warn" : "bad";
}

// Each row is [label, value, tone?]; rows with a null value are dropped.
function sections(rep) {
  const c = rep.capture, v = rep.coverage, g = rep.geometry, s = rep.space, f = rep.fidelity, a = rep.asset;
  const out = [];
  if (c) {
    out.push(["Capture", [
      ["Capture score", `${fmt(c.score)} / 100`, tone(c.score, 75, 50)],
      ["Footage", `${duration(c.duration_s)} in ${c.clips} clip${c.clips === 1 ? "" : "s"}`],
      ["Keyframes kept", c.keyframes.toLocaleString()],
      ["Blurred or too fast", `${fmt(c.blurry_pct + c.too_fast_pct, 1)}%`],
    ]]);
  }
  if (v) {
    out.push(["Surface seen well enough", [
      ["Walls, furniture, floor", `${fmt(v.walk_view_quality_pct)}%`, tone(v.walk_view_quality_pct, 90, 75)],
      ["Ceiling", `${fmt(v.ceiling_quality_pct)}%`, tone(v.ceiling_quality_pct, 70, 40)],
      ["Boundary closed", `${fmt(v.boundary_closure_pct)}%`, tone(v.boundary_closure_pct, 90, 70)],
      ["Median views per point", fmt(v.median_views)],
    ]]);
  }
  if (g) {
    const src = { input_is_metric: "metric model", marker: "printed marker", camera_height: "camera height",
                  given: "entered by hand", reference_model: "draft model" }[g.scale_source] || g.scale_source;
    const rows = [["Scale from", src]];
    if (g.model_scale_error_pct != null) {
      rows.push(["Draft model's own scale", `${g.model_scale_error_pct > 0 ? "+" : ""}${fmt(g.model_scale_error_pct, 1)}% off`]);
    }
    if (g.marker_error_pct != null && g.marker_independent) { // only meaningful when the marker did not set the scale
      const abs = Math.abs(g.marker_error_pct);
      rows.push(["Marker scale check", `${g.marker_error_pct > 0 ? "+" : ""}${fmt(g.marker_error_pct, 1)}%`,
                 abs <= 2 ? "ok" : abs <= 5 ? "warn" : "bad"]);
    }
    if (g.marker_sightings != null) rows.push(["Marker sightings", String(g.marker_sightings)]);
    rows.push(["Floor flatness", g.floor_rms_cm != null ? `${fmt(g.floor_rms_cm, 1)} cm RMS` : null]);
    rows.push(["Draft vs final poses", g.draft_vs_final_pose_rmse_cm != null ? `${fmt(g.draft_vs_final_pose_rmse_cm, 1)} cm RMS` : null]);
    out.push(["Geometry", rows]);
  }
  if (s) {
    out.push(["Space", [
      ["Floor area", s.floor_area_m2 != null ? `${fmt(s.floor_area_m2, 1)} m²` : null],
      ["Ceiling height", s.ceiling_height_m != null ? `${fmt(s.ceiling_height_m, 2)} m` : null],
      ["Capture path", s.capture_path_m != null ? `${fmt(s.capture_path_m, 1)} m` : null],
    ]]);
  }
  if (f) {
    out.push(["Fidelity on held-out views", [
      ["PSNR", f.psnr != null ? `${fmt(f.psnr, 2)} dB` : null],
      ["SSIM", fmt(f.ssim, 3)],
      ["LPIPS", fmt(f.lpips, 3)],
      ["Training", f.train_minutes != null ? `${fmt(f.train_minutes)} min, ${f.step.toLocaleString()} steps` : null],
    ]]);
  }
  if (a) {
    out.push(["Asset", [
      ["Gaussians", a.gaussians != null ? a.gaussians.toLocaleString() : null],
      ["Download", a.size_mb != null ? `${fmt(a.size_mb, 1)} MB` : null],
    ]]);
  }
  return out;
}

export class ReportCard {
  constructor(el, rep) {
    this.el = el;
    const parts = [`<h1>${escapeHtml(rep.title || "Scene")}</h1>`];
    if (rep.generated) parts.push(`<p class="sub">Twin report, built ${escapeHtml(rep.generated)}</p>`);
    for (const [name, rows] of sections(rep)) {
      const body = rows.filter((r) => r[1] != null)
        .map(([k, val, t]) => `<div class="row"><dt>${k}</dt><dd${t ? ` class="${t}"` : ""}>${escapeHtml(String(val))}</dd></div>`)
        .join("");
      if (body) parts.push(`<h2>${name}</h2><dl>${body}</dl>`);
    }
    parts.push(`<h2>On this device</h2><dl>
      <div class="row"><dt>Load time</dt><dd id="live-load">measuring</dd></div>
      <div class="row"><dt>Frame rate</dt><dd id="live-fps">measuring</dd></div>
      <div class="row"><dt>Render scale</dt><dd id="live-scale">100%</dd></div>
      <div class="row" id="live-bench-row" hidden><dt>Tour benchmark</dt><dd id="live-bench"></dd></div>
    </dl>`);
    el.innerHTML = parts.join("");
    this.load = el.querySelector("#live-load");
    this.fps = el.querySelector("#live-fps");
    this.scale = el.querySelector("#live-scale");
    this.benchRow = el.querySelector("#live-bench-row");
    this.bench = el.querySelector("#live-bench");
  }

  setLoad(ms) { this.load.textContent = `${(ms / 1000).toFixed(1)} s`; }

  setFps(fps) {
    this.fps.textContent = `${Math.round(fps)} fps`;
    this.fps.className = tone(fps, 45, 24);
  }

  setScale(s) { this.scale.textContent = `${Math.round(s * 100)}%`; }

  setBench(avg, low) {
    this.benchRow.hidden = false;
    this.bench.textContent = `${Math.round(avg)} fps average, ${Math.round(low)} fps worst 1%`;
  }
}

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
}
