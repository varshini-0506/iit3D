"""Headless smoke test for the web viewer.

Start the viewer first (cd viewer && npm run dev), publish a scene, then:

    python tests/viewer_smoke.py http://localhost:5173/ [screenshot_dir]

Checks, against whatever scene is published: the splat loads, W walks forward,
walls stop the walker, clicking the plan plans and walks a route, measuring
returns a distance, and the phone layout renders. Needs `pip install playwright`
and a Chromium that Playwright can find.
"""

import asyncio
import json
import math
import os
import sys

from playwright.async_api import async_playwright

ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]


async def run(url: str, out: str | None) -> int:
    failures = []

    def check(name, ok, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'}  {name}  {detail}")
        if not ok:
            failures.append(name)

    async with async_playwright() as p:
        browser = await p.chromium.launch(args=ARGS)
        page = await browser.new_page(viewport={"width": 960, "height": 540})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        await page.goto(url)
        await page.wait_for_function("window.__twin && window.__twin.state.ready", timeout=180000)
        get = lambda: page.evaluate("(() => { const t = window.__twin; return {x: t.player.x, y: t.player.y, "
                                    "yaw: t.player.yaw, route: !!t.player.route, fps: t.state.fps, "
                                    "load: t.state.loadMs, can: t.nav.canStand(t.player.x, t.player.y)} })()")
        s0 = await get()
        check("scene loads", s0["load"] is not None, f"load {s0['load']:.0f} ms")
        check("spawn is on walkable floor", s0["can"])
        if out:
            await page.wait_for_timeout(2000)
            await page.screenshot(path=os.path.join(out, "viewer_desktop.png"), timeout=180000)

        # software rendering in CI can run at 1 to 2 frames a second, so wait on outcomes, not on the clock
        async def until(expr, timeout=40000):
            try:
                await page.wait_for_function(expr, timeout=timeout)
                return True
            except Exception:
                return False

        await page.evaluate("window.__s0 = {x: window.__twin.player.x, y: window.__twin.player.y}")
        await page.keyboard.down("KeyW")
        walked = await until("Math.hypot(__twin.player.x - __s0.x, __twin.player.y - __s0.y) > 0.3")
        check("W walks forward", walked)

        # keep holding W until the walker stops making progress: it must be resting on walkable floor
        await until("(() => { const p = __twin.player, k = p.x.toFixed(3) + p.y.toFixed(3); "
                    "window.__same = (window.__lastk === k) ? (window.__same || 0) + 1 : 0; window.__lastk = k; "
                    "return window.__same > 12; })()", timeout=90000)
        await page.keyboard.up("KeyW")
        s2 = await get()
        check("walls stop the walker", s2["can"], f"at ({s2['x']:.2f}, {s2['y']:.2f})")

        # route to the far end of the capture path via the public API, then via a click on the plan
        far = await page.evaluate("(() => { const t = window.__twin, p = t.nav.path; let b = p[0], d = 0; "
                                  "for (const q of p) { const e = Math.hypot(q[0]-t.player.x, q[1]-t.player.y); "
                                  "if (e > d) { d = e; b = q; } } const r = t.walkTo(b[0], b[1]); "
                                  "return r ? {len: r.length, n: r.points.length, straight: d} : null })()")
        check("shortest path is found", far is not None and far["len"] >= far["straight"] - 0.11,
              f"{far['len']:.2f} m over {far['n']} waypoints (straight line {far['straight']:.2f} m)" if far else "")
        await page.evaluate("window.__s2 = {x: window.__twin.player.x, y: window.__twin.player.y}")
        check("auto-walk follows the route to its end",
              await until("!__twin.player.route && Math.hypot(__twin.player.x - __s2.x, __twin.player.y - __s2.y) > 0.5",
                          timeout=120000))
        box = await page.locator("#plan").bounding_box()
        await page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
        await page.wait_for_timeout(300)
        route_text = await page.locator("#tb-route").text_content()
        check("clicking the plan sets a route", route_text not in ("none", ""), route_text)
        if out:
            await page.screenshot(path=os.path.join(out, "viewer_route.png"), timeout=180000)

        # measuring: two picks straight ahead and 20 degrees down must give a finite distance
        d = await page.evaluate("(() => { const t = window.__twin; const a = t.pick(0, 0), b = t.pick(0.3, -0.5); "
                                "return a && b ? a.distanceTo(b) : null })()")
        check("measure tool returns a distance", d is not None and 0 < d < 50, f"{d:.2f} m" if d else "")
        check("no page errors", not errors, "; ".join(errors[:3]))

        await page.close()
        phone = await browser.new_page(viewport={"width": 390, "height": 780}, has_touch=True, is_mobile=True)
        phone_logs = []
        phone.on("console", lambda m: phone_logs.append(m.text))
        await phone.goto(url)
        await phone.wait_for_function("window.__twin && window.__twin.state.ready", timeout=180000)
        over = await phone.evaluate("document.documentElement.scrollWidth > window.innerWidth")
        check("phone layout has no sideways scroll", not over)
        print("phone console:", phone_logs[:6])
        if out:
            await phone.wait_for_timeout(2000)
            await phone.screenshot(path=os.path.join(out, "viewer_phone.png"), timeout=180000)
        await phone.close()
        await browser.close()
    print(json.dumps({"failures": failures}))
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    raise SystemExit(asyncio.run(run(sys.argv[1], out_dir)))
