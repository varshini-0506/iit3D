// The plan sheet: a live floor plan drawn from the nav grid, in the manner of a drafting sheet.
// Click anywhere on it to walk there by the shortest route.

const INK = "#1b2a3a";
const PAPER = "#f3f5f2";
const UNKNOWN = "#e2e6e3";
const FURNITURE = "#b4bcc0";
const PATH = "#2f6f9f";
const MARK = "#c8372d";

export class PlanSheet {
  constructor(canvas, nav, { onPick }) {
    this.canvas = canvas;
    this.nav = nav;
    this.ctx = canvas.getContext("2d");
    this.showCoverage = false;
    this.coverage = null;      // { image, grid }
    this.base = this._renderBase();
    this.resize();
    canvas.addEventListener("click", (e) => {
      const r = canvas.getBoundingClientRect();
      const [x, y] = this.toWorld(e.clientX - r.left, e.clientY - r.top);
      onPick(x, y);
    });
  }

  // one offscreen pixel per nav cell, +Y up
  _renderBase() {
    const { w, h, free, wall, obstacle } = this.nav;
    const c = new OffscreenCanvas(w, h);
    const ctx = c.getContext("2d");
    const img = ctx.createImageData(w, h);
    const put = (i, hex) => {
      img.data[i] = parseInt(hex.slice(1, 3), 16);
      img.data[i + 1] = parseInt(hex.slice(3, 5), 16);
      img.data[i + 2] = parseInt(hex.slice(5, 7), 16);
      img.data[i + 3] = 255;
    };
    for (let r = 0; r < h; r++) {
      for (let col = 0; col < w; col++) {
        const src = r * w + col;
        const dst = ((h - 1 - r) * w + col) * 4;
        put(dst, wall[src] ? INK : obstacle[src] ? FURNITURE : free[src] ? PAPER : UNKNOWN);
      }
    }
    ctx.putImageData(img, 0, 0);
    return c;
  }

  setCoverage(image, grid) { this.coverage = { image, grid }; }

  resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const cssW = this.canvas.clientWidth || 260;
    const cssH = Math.round(cssW * this.nav.h / this.nav.w);
    this.canvas.style.height = cssH + "px";
    this.canvas.width = Math.round(cssW * dpr);
    this.canvas.height = Math.round(cssH * dpr);
    this.scale = this.canvas.width / this.nav.w; // canvas pixels per cell
    this.cssScale = cssW / this.nav.w;
  }

  toCanvas(x, y) {
    const n = this.nav;
    return [(x - n.x0) / n.cell * this.scale, this.canvas.height - (y - n.y0) / n.cell * this.scale];
  }

  toWorld(cssX, cssY) {
    const n = this.nav;
    const cssH = this.canvas.clientHeight;
    return [n.x0 + cssX / this.cssScale * n.cell, n.y0 + (cssH - cssY) / this.cssScale * n.cell];
  }

  draw(player) {
    const ctx = this.ctx, W = this.canvas.width, H = this.canvas.height;
    const px = Math.max(1, this.scale / 3);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.base, 0, 0, W, H);

    if (this.showCoverage && this.coverage) {
      const g = this.coverage.grid, n = this.nav;
      const [x0, y1] = this.toCanvas(g.x0, g.y0);
      const wpx = g.width * g.cell / n.cell * this.scale, hpx = g.height * g.cell / n.cell * this.scale;
      ctx.save();
      ctx.globalAlpha = 0.85;
      ctx.translate(x0, y1);
      ctx.scale(1, -1); // the overlay's row 0 is the lowest y
      ctx.drawImage(this.coverage.image, 0, 0, wpx, hpx);
      ctx.restore();
    }

    // the path the camera took during capture
    const path = this.nav.path;
    if (path.length > 1) {
      ctx.strokeStyle = PATH;
      ctx.lineWidth = px;
      ctx.globalAlpha = 0.55;
      ctx.beginPath();
      for (let i = 0; i < path.length; i++) {
        const [cx, cy] = this.toCanvas(path[i][0], path[i][1]);
        const gap = i > 0 && Math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1]) > 1.5;
        if (i === 0 || gap) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // the route being walked
    const route = player.route;
    if (route) {
      ctx.strokeStyle = MARK;
      ctx.lineWidth = px * 1.6;
      ctx.setLineDash([px * 4, px * 3]);
      ctx.beginPath();
      const [sx, sy] = this.toCanvas(player.x, player.y);
      ctx.moveTo(sx, sy);
      for (let i = route.i; i < route.points.length; i++) {
        const [cx, cy] = this.toCanvas(route.points[i][0], route.points[i][1]);
        ctx.lineTo(cx, cy);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      const end = route.points[route.points.length - 1];
      const [ex, ey] = this.toCanvas(end[0], end[1]);
      ctx.beginPath();
      ctx.arc(ex, ey, px * 3, 0, Math.PI * 2);
      ctx.stroke();
    }

    // the viewer: a view cone and a dot
    const [cx, cy] = this.toCanvas(player.x, player.y);
    const a = -player.yaw; // canvas y points down
    const reach = 1.6 / this.nav.cell * this.scale;
    ctx.fillStyle = "rgba(200, 55, 45, 0.18)";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, reach, a - 0.6, a + 0.6);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = MARK;
    ctx.beginPath();
    ctx.arc(cx, cy, px * 3.2, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = PAPER;
    ctx.lineWidth = px;
    ctx.stroke();
  }
}
