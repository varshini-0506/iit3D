// NammaTwin web viewer: walk a metric Gaussian-splat twin in the browser.
//
//   ?scene=<folder>   load a different published scene (default: scene/)
//   ?bench=1          replay the capture path once and report the frame rate
//
// The scene folder is written by `python -m nammatwin report`: manifest.json,
// the splat (.sog / .spz / .ply), nav.json + nav.png, and report.json.

import "@fontsource/barlow-semi-condensed/400.css";
import "@fontsource/barlow-semi-condensed/500.css";
import "@fontsource/barlow-semi-condensed/600.css";
import "./style.css";

import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";

import { ReportCard } from "./card.js";
import { Player } from "./controls.js";
import { NavGrid } from "./nav.js";
import { PlanSheet } from "./plan.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const sceneBase = new URL((params.get("scene") || "scene").replace(/\/?$/, "/"), location.href);
const isTouch = matchMedia("(pointer: coarse)").matches;
const t0 = performance.now();

function setProgress(text, frac) {
  $("loading-text").textContent = text;
  if (frac != null) $("loading-fill").style.width = `${Math.round(Math.min(1, frac) * 100)}%`;
}

function fail(message) {
  $("loading").classList.add("error");
  $("loading").hidden = false;
  $("loading-text").textContent = message;
}

async function getJson(name) {
  const url = new URL(name, sceneBase);
  let res;
  try { res = await fetch(url); } catch { res = null; }
  if (!res || !res.ok) throw new Error(`missing:${name}`);
  return res.json();
}

async function boot() {
  let manifest, navMeta, report = {};
  try {
    manifest = await getJson("manifest.json");
    navMeta = await getJson(manifest.nav || "nav.json");
    if (manifest.report) report = await getJson(manifest.report);
  } catch (e) {
    const name = String(e.message).replace("missing:", "");
    fail(`No scene found: ${name} is missing from ${sceneBase.pathname}\n\n` +
         "Publish one with:\n  python -m nammatwin sample --out work --scene viewer/public/scene\n" +
         "or finish a real capture with scripts/run_pipeline.sh.");
    return;
  }
  if (!manifest.splat) {
    fail("This scene has a floor plan but no splat file. Run the packaging step, then `python -m nammatwin report --splat ...`.");
    return;
  }
  document.title = `${manifest.title || "Scene"} | NammaTwin`;
  const nav = await NavGrid.load(sceneBase, navMeta);

  // ---- renderer ---------------------------------------------------------------
  const canvas = $("view");
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance" });
  } catch {
    fail("This browser could not start WebGL 2, which the viewer needs. Try a current Chrome, Edge, Firefox or Safari.");
    return;
  }
  canvas.addEventListener("webglcontextlost", (e) => {
    e.preventDefault();
    console.warn("WebGL context lost");
    fail("The browser took the graphics context away, usually because the device ran short of graphics memory.\nReload the page to carry on.");
  });
  const basePixelRatio = Math.min(devicePixelRatio || 1, 2);
  let renderScale = 1;
  renderer.setPixelRatio(basePixelRatio);
  renderer.setSize(innerWidth, innerHeight, false);
  renderer.setClearColor(0x0e1820);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(72, innerWidth / innerHeight, 0.05, 300);
  camera.up.set(0, 0, 1); // the pipeline delivers Z-up, metric scenes
  const spark = new SparkRenderer({ renderer });
  scene.add(spark);

  // ---- the splat ----------------------------------------------------------------
  setProgress("Downloading the scene", 0.02);
  const gaussians = report.asset?.gaussians ?? 0;
  const splat = new SplatMesh({
    url: new URL(manifest.splat, sceneBase).href,
    lod: manifest.lod ?? gaussians > 1_500_000, // build levels of detail for big scenes
    onProgress: (e) => {
      if (e.lengthComputable) setProgress("Downloading the scene", 0.02 + 0.9 * (e.loaded / e.total));
    },
  });
  try {
    await splat.initialized;
  } catch (e) {
    fail(`Could not read ${manifest.splat}.\n${e?.message || e}`);
    return;
  }
  scene.add(splat);
  setProgress("Preparing the view", 0.97);

  // ---- people-facing parts --------------------------------------------------------
  const player = new Player(nav, canvas);
  const card = new ReportCard($("card"), { ...report, title: manifest.title || report.title });
  const plan = new PlanSheet($("plan"), nav, {
    onPick: (x, y) => {
      const r = player.walkTo(x, y);
      $("tb-route").textContent = r ? `${r.length.toFixed(1)} m` : "no way through";
    },
  });
  $("tb-title").textContent = manifest.title || "Scene";
  $("tb-area").textContent = nav.stats.floor_area_m2 != null ? `${nav.stats.floor_area_m2.toFixed(1)} m²` : "n/a";
  const placeScaleBar = () => { $("tb-bar").style.width = `${Math.max(8, plan.cssScale / nav.cell)}px`; };

  if (manifest.coverage) {
    const img = new Image();
    img.onload = () => plan.setCoverage(img, manifest.coverage_grid || navMeta);
    img.src = new URL(manifest.coverage, sceneBase).href;
  } else {
    $("btn-coverage").hidden = true;
  }

  // ---- measuring -------------------------------------------------------------------
  const measure = { on: false, pts: [], group: new THREE.Group() };
  scene.add(measure.group);
  const markMat = new THREE.MeshBasicMaterial({ color: 0xc8372d, depthTest: false });
  const lineMat = new THREE.LineBasicMaterial({ color: 0xc8372d, depthTest: false });
  const raycaster = new THREE.Raycaster();
  const floor = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);

  function pick(ndcX, ndcY) {
    raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), camera);
    try {
      const hits = raycaster.intersectObject(splat, false).filter((h) => h.distance > 0.15);
      if (hits.length) return hits.sort((a, b) => a.distance - b.distance)[0].point.clone();
    } catch { /* fall through to the floor plane */ }
    const p = new THREE.Vector3();
    return raycaster.ray.intersectPlane(floor, p) ? p : null;
  }

  function addMeasurePoint(p) {
    if (measure.pts.length >= 2) { measure.pts = []; measure.group.clear(); $("measure").hidden = true; }
    measure.pts.push(p);
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.02, 12, 8), markMat);
    dot.position.copy(p);
    dot.renderOrder = 10;
    measure.group.add(dot);
    if (measure.pts.length === 2) {
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(measure.pts), lineMat);
      line.renderOrder = 10;
      measure.group.add(line);
      const d = measure.pts[0].distanceTo(measure.pts[1]);
      $("measure").textContent = d >= 1 ? `${d.toFixed(2)} m` : `${(d * 100).toFixed(1)} cm`;
      $("measure").hidden = false;
    }
  }

  function placeMeasureLabel() {
    if (measure.pts.length !== 2) return;
    const mid = measure.pts[0].clone().add(measure.pts[1]).multiplyScalar(0.5).project(camera);
    const el = $("measure");
    el.style.visibility = mid.z > 1 ? "hidden" : "visible";
    el.style.left = `${(mid.x * 0.5 + 0.5) * innerWidth}px`;
    el.style.top = `${(-mid.y * 0.5 + 0.5) * innerHeight}px`;
  }

  // ---- tools -----------------------------------------------------------------------
  const press = (id, on) => $(id).setAttribute("aria-pressed", String(on));
  const toggleFly = () => { player.setFly(!player.fly); press("btn-fly", player.fly); };
  const toggleMeasure = () => {
    measure.on = !measure.on;
    press("btn-measure", measure.on);
    canvas.classList.toggle("measuring", measure.on);
    $("reticle").hidden = !(measure.on && player.locked);
    if (!measure.on) { measure.pts = []; measure.group.clear(); $("measure").hidden = true; }
  };
  const toggleTour = () => {
    if (player.tour) player.tour = null; else player.startTour();
    press("btn-fly", player.fly);
  };
  const toggleCoverage = () => { plan.showCoverage = !plan.showCoverage; press("btn-coverage", plan.showCoverage); };
  const toggleReport = () => { $("card").hidden = !$("card").hidden; press("btn-report", !$("card").hidden); };
  $("btn-fly").onclick = toggleFly;
  $("btn-measure").onclick = toggleMeasure;
  $("btn-tour").onclick = toggleTour;
  $("btn-coverage").onclick = toggleCoverage;
  $("btn-report").onclick = toggleReport;
  addEventListener("keydown", (e) => {
    if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.code === "KeyF") toggleFly();
    else if (e.code === "KeyM") toggleMeasure();
    else if (e.code === "KeyT") toggleTour();
    else if (e.code === "KeyC" && manifest.coverage) toggleCoverage();
    else if (e.code === "KeyR") toggleReport();
  });

  // a click either takes a measurement or captures the mouse for looking around
  let downAt = null;
  canvas.addEventListener("mousedown", (e) => { downAt = [e.clientX, e.clientY]; });
  canvas.addEventListener("click", (e) => {
    const still = downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) < 5;
    if (measure.on) {
      if (!still && !player.locked) return;
      const p = player.locked ? pick(0, 0) : pick((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
      if (p) addMeasurePoint(p);
    } else if (!isTouch && !player.locked && still) {
      canvas.requestPointerLock?.();
    }
  });
  canvas.addEventListener("touchend", (e) => {
    if (!measure.on || e.changedTouches.length !== 1) return;
    const t = e.changedTouches[0];
    const p = pick((t.clientX / innerWidth) * 2 - 1, -(t.clientY / innerHeight) * 2 + 1);
    if (p) addMeasurePoint(p);
  });
  document.addEventListener("pointerlockchange", () => {
    canvas.classList.toggle("locked", player.locked);
    $("reticle").hidden = !(measure.on && player.locked);
  });

  const hint = $("hint");
  hint.textContent = isTouch
    ? "Drag the left side to walk and the right side to look. Tap the plan to walk to a spot."
    : "Click the view to look around, then W A S D to walk. Click the plan to walk to a spot. Esc frees the mouse.";

  addEventListener("resize", () => {
    renderer.setSize(innerWidth, innerHeight, false);
    camera.aspect = innerWidth / innerHeight;
    camera.updateProjectionMatrix();
    plan.resize();
    placeScaleBar();
  });

  // ---- go ----------------------------------------------------------------------------
  for (const id of ["tools", "card", "sheet", "hint"]) $(id).hidden = false;
  if (innerWidth < 760) toggleReport(); // keep the small screen clear; the Report button brings it back
  plan.resize();
  placeScaleBar();

  const bench = params.get("bench") ? { times: [], done: false, started: false } : null;

  const state = { fps: 0, loadMs: null, renderScale: 1, frames: 0, splats: gaussians, bench: null, ready: false };
  window.__twin = { player, nav, plan, state, camera, splat, pick, walkTo: (x, y) => player.walkTo(x, y) };

  let last = performance.now(), acc = 0, count = 0, slowFor = 0, fastFor = 0, shownFrames = 0;
  const sceneAdded = performance.now();
  renderer.setAnimationLoop((now) => {
    const realDt = (now - last) / 1000;
    const dt = Math.min(0.25, realDt); // a stalled tab must not teleport the walker
    last = now;
    player.update(dt);
    player.apply(camera);
    renderer.render(scene, camera);
    plan.draw(player);
    placeMeasureLabel();
    if (player.moved || player.route || player.tour) hint.classList.add("gone");

    state.frames++;
    if (state.loadMs == null) {
      // loading ends when splat pixels are actually on screen, not when the file has arrived
      if (splatsVisible()) shownFrames++;
      if (shownFrames >= 2 || now - sceneAdded > 15000) {
        state.loadMs = now - t0;
        state.ready = true;
        card.setLoad(state.loadMs);
        $("loading").hidden = true;
      }
    }
    if (realDt < 1) { acc += realDt; count++; } // ignore stalls such as a hidden tab
    if (bench && state.ready && !bench.started) { bench.started = true; player.startTour(); }
    if (bench && bench.started && !bench.done) {
      bench.times.push(realDt);
      if (!player.tour && bench.times.length > 10) {
        bench.done = true;
        const sorted = bench.times.slice().sort((a, b) => b - a);
        const avg = bench.times.length / bench.times.reduce((a, b) => a + b, 0);
        const low = 1 / sorted[Math.floor(sorted.length * 0.01)];
        state.bench = { fps_avg: avg, fps_low_1pct: low, frames: bench.times.length };
        card.setBench(avg, low);
      }
    }
    if (acc >= 0.5) {
      state.fps = count / acc;
      card.setFps(state.fps);
      // adaptive resolution: give up pixels before giving up smooth motion
      if (!state.ready) { slowFor = fastFor = 0; } else if (state.fps < 24) { slowFor += acc; fastFor = 0; } else if (state.fps > 55) { fastFor += acc; slowFor = 0; } else { slowFor = fastFor = 0; }
      if (slowFor > 2 && renderScale > 0.5) { renderScale = Math.max(0.5, renderScale * 0.8); slowFor = 0; applyScale(); }
      if (fastFor > 6 && renderScale < 1) { renderScale = Math.min(1, renderScale / 0.9); fastFor = 0; applyScale(); }
      acc = 0; count = 0;
    }
  });

  // Sample a sparse grid of the frame just drawn; anything other than the clear colour means splats.
  function splatsVisible() {
    const gl = renderer.getContext();
    const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight;
    const px = new Uint8Array(4);
    const packBuffer = gl.getParameter(gl.PIXEL_PACK_BUFFER_BINDING); // the splat renderer may have one bound
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    let seen = false;
    for (let j = 1; j <= 4 && !seen; j++) {
      for (let i = 1; i <= 4 && !seen; i++) {
        gl.readPixels(Math.floor(w * i / 5), Math.floor(h * j / 5), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
        seen = Math.abs(px[0] - 0x0e) + Math.abs(px[1] - 0x18) + Math.abs(px[2] - 0x20) > 12;
      }
    }
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, packBuffer);
    return seen;
  }

  function applyScale() {
    renderer.setPixelRatio(basePixelRatio * renderScale);
    renderer.setSize(innerWidth, innerHeight, false);
    state.renderScale = renderScale;
    card.setScale(renderScale);
  }
}

boot().catch((e) => {
  console.error(e);
  fail(`The viewer stopped while starting up.\n${e?.message || e}`);
});
