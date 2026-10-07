// First-person movement in a Z-up, metric world.
// Walk mode keeps the eyes at a fixed real height and collides with the nav grid.
// Fly mode ignores both, for looking at the twin from outside.

const WALK_SPEED = 1.4;   // m/s, a relaxed indoor pace
const RUN_FACTOR = 2.2;
const LOOK_RATE = 0.0022; // radians per pixel of mouse travel
const MAX_PITCH = 1.45;

export class Player {
  constructor(nav, canvas) {
    this.nav = nav;
    this.canvas = canvas;
    this.eye = nav.spawn.eye_height ?? 1.6;
    const s = nav.snap(nav.spawn.x, nav.spawn.y) || [nav.spawn.x, nav.spawn.y];
    this.x = s[0];
    this.y = s[1];
    this.z = this.eye;
    this.yaw = nav.spawn.yaw ?? 0;
    this.pitch = 0;
    this.fly = false;
    this.keys = new Set();
    this.route = null;      // { points, length, i } while auto-walking
    this.tour = null;       // { points, i } while replaying the capture path
    this.moved = false;     // has the person moved yet (used to fade the hint)
    this.travelled = 0;
    this.stick = { x: 0, y: 0 };
    this.onInput = () => {};
    this._bind();
  }

  get locked() { return document.pointerLockElement === this.canvas; }

  _bind() {
    const c = this.canvas;
    addEventListener("keydown", (e) => {
      if (e.target instanceof HTMLInputElement) return;
      this.keys.add(e.code);
      if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space"].includes(e.code)) e.preventDefault();
    });
    addEventListener("keyup", (e) => this.keys.delete(e.code));
    addEventListener("blur", () => this.keys.clear());

    // mouse: pointer lock when the browser grants it, drag-to-look otherwise
    let dragging = false, lx = 0, ly = 0;
    c.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      dragging = true; lx = e.clientX; ly = e.clientY;
    });
    addEventListener("mouseup", () => { dragging = false; });
    addEventListener("mousemove", (e) => {
      let dx = 0, dy = 0;
      if (this.locked) { dx = e.movementX; dy = e.movementY; }
      else if (dragging) { dx = lx - e.clientX; dy = ly - e.clientY; lx = e.clientX; ly = e.clientY; }
      else return;
      this.look(-dx * LOOK_RATE, -dy * LOOK_RATE);
    });

    // touch: left half of the screen is a movement stick, right half looks around
    const touches = new Map();
    c.addEventListener("touchstart", (e) => {
      for (const t of e.changedTouches) {
        touches.set(t.identifier, { sx: t.clientX, sy: t.clientY, x: t.clientX, y: t.clientY,
                                    move: t.clientX < innerWidth / 2 });
      }
      e.preventDefault();
    }, { passive: false });
    c.addEventListener("touchmove", (e) => {
      for (const t of e.changedTouches) {
        const s = touches.get(t.identifier);
        if (!s) continue;
        if (s.move) {
          this.stick.x = clamp((t.clientX - s.sx) / 60, -1, 1);
          this.stick.y = clamp((s.sy - t.clientY) / 60, -1, 1);
        } else {
          this.look(-(t.clientX - s.x) * LOOK_RATE * 1.6, -(t.clientY - s.y) * LOOK_RATE * 1.6);
        }
        s.x = t.clientX; s.y = t.clientY;
      }
      e.preventDefault();
    }, { passive: false });
    const end = (e) => {
      for (const t of e.changedTouches) {
        const s = touches.get(t.identifier);
        if (s && s.move) { this.stick.x = 0; this.stick.y = 0; }
        touches.delete(t.identifier);
      }
    };
    c.addEventListener("touchend", end);
    c.addEventListener("touchcancel", end);
  }

  look(dyaw, dpitch) {
    this.yaw += dyaw;
    this.pitch = clamp(this.pitch + dpitch, -MAX_PITCH, MAX_PITCH);
    if (this.tour) this.tour = null;
    this.onInput();
  }

  setFly(on) {
    this.fly = on;
    if (!on) {
      const s = this.nav.snap(this.x, this.y, 50);
      if (s) { this.x = s[0]; this.y = s[1]; }
      this.z = this.eye;
    }
  }

  walkTo(x, y) {
    const r = this.nav.findPath(this.x, this.y, x, y);
    if (!r) return null;
    this.fly = false;
    this.z = this.eye;
    this.tour = null;
    this.route = { ...r, i: 1 };
    return r;
  }

  startTour() {
    const pts = this.nav.path;
    if (pts.length < 2) return false;
    this.fly = false;
    this.route = null;
    this.z = this.eye;
    this.tour = { points: pts, i: 1 };
    this.x = pts[0][0];
    this.y = pts[0][1];
    return true;
  }

  update(dt) {
    const k = this.keys;
    let f = (k.has("KeyW") || k.has("ArrowUp") ? 1 : 0) - (k.has("KeyS") || k.has("ArrowDown") ? 1 : 0) + this.stick.y;
    let s = (k.has("KeyD") ? 1 : 0) - (k.has("KeyA") ? 1 : 0) + this.stick.x;
    const turn = (k.has("ArrowLeft") ? 1 : 0) - (k.has("ArrowRight") ? 1 : 0);
    if (turn) this.yaw += turn * 1.8 * dt;
    const manual = f !== 0 || s !== 0;
    if (manual) { this.route = null; this.tour = null; this.moved = true; this.onInput(); }

    const px = this.x, py = this.y;
    if (manual) {
      const len = Math.hypot(f, s);
      if (len > 1) { f /= len; s /= len; }
      const speed = WALK_SPEED * (k.has("ShiftLeft") || k.has("ShiftRight") ? RUN_FACTOR : 1);
      const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw);
      // forward is (cos yaw, sin yaw); right is (sin yaw, -cos yaw)
      const dx = (cy * f + sy * s) * speed * dt;
      const dy = (sy * f - cy * s) * speed * dt;
      if (this.fly) {
        const cp = Math.cos(this.pitch);
        this.x += cy * f * cp * speed * dt + sy * s * speed * dt;
        this.y += sy * f * cp * speed * dt - cy * s * speed * dt;
        this.z += Math.sin(this.pitch) * f * speed * dt;
      } else {
        [this.x, this.y] = this.nav.move(this.x, this.y, dx, dy);
      }
    } else if (this.route) {
      this._follow(this.route, WALK_SPEED * 0.95, dt, false);
      if (this.route && this.route.i >= this.route.points.length) this.route = null;
    } else if (this.tour) {
      this._follow(this.tour, WALK_SPEED * 0.6, dt, true);
      if (this.tour && this.tour.i >= this.tour.points.length) this.tour = null;
    }
    if (this.fly) {
      const v = (k.has("KeyE") || k.has("Space") ? 1 : 0) - (k.has("KeyQ") ? 1 : 0);
      this.z += v * WALK_SPEED * dt;
    }
    this.travelled += Math.hypot(this.x - px, this.y - py);
  }

  _follow(job, speed, dt, jumpGaps) {
    let budget = speed * dt;
    while (budget > 0 && job.i < job.points.length) {
      const [tx, ty] = job.points[job.i];
      const dx = tx - this.x, dy = ty - this.y;
      const d = Math.hypot(dx, dy);
      if (jumpGaps && d > 1.5) { this.x = tx; this.y = ty; job.i++; continue; } // gap between clips
      if (d > 1e-4) {
        const want = Math.atan2(dy, dx);
        let diff = want - this.yaw;
        diff = Math.atan2(Math.sin(diff), Math.cos(diff));
        this.yaw += diff * Math.min(1, dt * 3.5);
      }
      if (d <= budget) { this.x = tx; this.y = ty; budget -= d; job.i++; }
      else { this.x += (dx / d) * budget; this.y += (dy / d) * budget; budget = 0; }
    }
    this.pitch += (-0.08 - this.pitch) * Math.min(1, dt * 2);
  }

  apply(camera) {
    camera.position.set(this.x, this.y, this.z);
    const cp = Math.cos(this.pitch);
    camera.up.set(0, 0, 1);
    camera.lookAt(this.x + Math.cos(this.yaw) * cp, this.y + Math.sin(this.yaw) * cp, this.z + Math.sin(this.pitch));
  }
}

function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }
