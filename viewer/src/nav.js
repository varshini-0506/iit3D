// The walkable grid produced by `python -m nammatwin nav`.
// One 2D grid does three jobs here: collision for walking, the plan view,
// and shortest-path routing (the same map Round 2 navigates on).

const SQRT2 = Math.SQRT2;

export class NavGrid {
  constructor(meta, pixels) {
    this.cell = meta.cell;
    this.x0 = meta.x0;
    this.y0 = meta.y0;
    this.w = meta.width;
    this.h = meta.height;
    this.spawn = meta.spawn;
    this.path = meta.path || [];
    this.stats = meta.stats || {};
    const n = this.w * this.h;
    this.walk = new Uint8Array(n);
    this.wall = new Uint8Array(n);
    this.obstacle = new Uint8Array(n);
    this.free = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const r = pixels[4 * i], g = pixels[4 * i + 1], b = pixels[4 * i + 2];
      this.walk[i] = r > 127 ? 1 : 0;
      this.wall[i] = g > 127 ? 1 : 0;
      this.obstacle[i] = b > 200 ? 1 : 0;
      this.free[i] = b > 64 && b <= 200 ? 1 : 0;
    }
  }

  static async load(baseUrl, meta) {
    const res = await fetch(new URL(meta.image, baseUrl));
    if (!res.ok) throw new Error(`Could not load ${meta.image} (${res.status})`);
    const bmp = await createImageBitmap(await res.blob(), {
      colorSpaceConversion: "none",
      premultiplyAlpha: "none",
    });
    if (bmp.width !== meta.width || bmp.height !== meta.height) {
      throw new Error(`nav image is ${bmp.width}x${bmp.height} but nav.json says ${meta.width}x${meta.height}`);
    }
    const c = new OffscreenCanvas(bmp.width, bmp.height);
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bmp, 0, 0);
    return new NavGrid(meta, ctx.getImageData(0, 0, bmp.width, bmp.height).data);
  }

  col(x) { return Math.floor((x - this.x0) / this.cell); }
  row(y) { return Math.floor((y - this.y0) / this.cell); }
  cx(c) { return this.x0 + (c + 0.5) * this.cell; }
  cy(r) { return this.y0 + (r + 0.5) * this.cell; }

  canStand(x, y) {
    const c = this.col(x), r = this.row(y);
    return c >= 0 && r >= 0 && c < this.w && r < this.h && this.walk[r * this.w + c] === 1;
  }

  // Move with wall sliding. Sub-steps keep a slow frame from tunnelling through a thin wall.
  move(x, y, dx, dy) {
    const dist = Math.hypot(dx, dy);
    const steps = Math.max(1, Math.ceil(dist / (this.cell * 0.4)));
    const sx = dx / steps, sy = dy / steps;
    for (let i = 0; i < steps; i++) {
      if (this.canStand(x + sx, y + sy)) { x += sx; y += sy; }
      else if (this.canStand(x + sx, y)) { x += sx; }
      else if (this.canStand(x, y + sy)) { y += sy; }
      else break;
    }
    return [x, y];
  }

  // Nearest standable cell centre, searching outward in rings.
  snap(x, y, maxRadius = 3.0) {
    if (this.canStand(x, y)) return [x, y];
    const c0 = this.col(x), r0 = this.row(y);
    const maxR = Math.ceil(maxRadius / this.cell);
    let best = null, bestD = Infinity;
    for (let rad = 1; rad <= maxR; rad++) {
      for (let dr = -rad; dr <= rad; dr++) {
        for (let dc = -rad; dc <= rad; dc++) {
          if (Math.max(Math.abs(dr), Math.abs(dc)) !== rad) continue;
          const c = c0 + dc, r = r0 + dr;
          if (c < 0 || r < 0 || c >= this.w || r >= this.h || !this.walk[r * this.w + c]) continue;
          const d = dc * dc + dr * dr;
          if (d < bestD) { bestD = d; best = [this.cx(c), this.cy(r)]; }
        }
      }
      if (best && rad * rad >= bestD) return best;
    }
    return best;
  }

  lineOfSight(ax, ay, bx, by) {
    const n = Math.max(2, Math.ceil(Math.hypot(bx - ax, by - ay) / (this.cell * 0.4)));
    for (let i = 0; i <= n; i++) {
      const t = i / n;
      if (!this.canStand(ax + (bx - ax) * t, ay + (by - ay) * t)) return false;
    }
    return true;
  }

  // A* over the walk mask (8-connected, no corner cutting), then string-pulling.
  // Returns { points: [[x, y], ...], length } in metres, or null when unreachable.
  findPath(ax, ay, bx, by) {
    const a = this.snap(ax, ay), b = this.snap(bx, by);
    if (!a || !b) return null;
    const w = this.w, h = this.h, walk = this.walk;
    const start = this.row(a[1]) * w + this.col(a[0]);
    const goal = this.row(b[1]) * w + this.col(b[0]);
    const gc = goal % w, gr = (goal - gc) / w;
    const g = new Float32Array(w * h).fill(Infinity);
    const from = new Int32Array(w * h).fill(-1);
    const closed = new Uint8Array(w * h);
    const heap = new MinHeap();
    const hfn = (i) => {
      const c = i % w, r = (i - c) / w;
      const dx = Math.abs(c - gc), dy = Math.abs(r - gr);
      return Math.max(dx, dy) + (SQRT2 - 1) * Math.min(dx, dy);
    };
    g[start] = 0;
    heap.push(start, hfn(start));
    let found = false;
    while (heap.size) {
      const cur = heap.pop();
      if (closed[cur]) continue;
      closed[cur] = 1;
      if (cur === goal) { found = true; break; }
      const c = cur % w, r = (cur - c) / w;
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          if (!dr && !dc) continue;
          const nc = c + dc, nr = r + dr;
          if (nc < 0 || nr < 0 || nc >= w || nr >= h) continue;
          const ni = nr * w + nc;
          if (!walk[ni] || closed[ni]) continue;
          if (dr && dc && (!walk[r * w + nc] || !walk[nr * w + c])) continue;
          const cost = g[cur] + (dr && dc ? SQRT2 : 1);
          if (cost < g[ni]) { g[ni] = cost; from[ni] = cur; heap.push(ni, cost + hfn(ni)); }
        }
      }
    }
    if (!found) return null;
    const cells = [];
    for (let i = goal; i !== -1; i = from[i]) cells.push(i);
    cells.reverse();
    let pts = cells.map((i) => [this.cx(i % w), this.cy(Math.floor(i / w))]);
    pts[0] = a;
    pts[pts.length - 1] = b;
    // string pulling: skip waypoints that can be reached in a straight line
    const out = [pts[0]];
    let i = 0;
    while (i < pts.length - 1) {
      let j = pts.length - 1;
      while (j > i + 1 && !this.lineOfSight(pts[i][0], pts[i][1], pts[j][0], pts[j][1])) j--;
      out.push(pts[j]);
      i = j;
    }
    let length = 0;
    for (let k = 1; k < out.length; k++) length += Math.hypot(out[k][0] - out[k - 1][0], out[k][1] - out[k - 1][1]);
    return { points: out, length };
  }
}

class MinHeap {
  constructor() { this.ids = []; this.keys = []; }
  get size() { return this.ids.length; }
  push(id, key) {
    const ids = this.ids, keys = this.keys;
    let i = ids.length;
    ids.push(id); keys.push(key);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (keys[p] <= key) break;
      ids[i] = ids[p]; keys[i] = keys[p];
      i = p;
    }
    ids[i] = id; keys[i] = key;
  }
  pop() {
    const ids = this.ids, keys = this.keys;
    const top = ids[0];
    const lastId = ids.pop(), lastKey = keys.pop();
    const n = ids.length;
    if (n) {
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= n) break;
        if (c + 1 < n && keys[c + 1] < keys[c]) c++;
        if (keys[c] >= lastKey) break;
        ids[i] = ids[c]; keys[i] = keys[c];
        i = c;
      }
      ids[i] = lastId; keys[i] = lastKey;
    }
    return top;
  }
}
