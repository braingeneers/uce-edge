"""Playwright driver for Phase 2 — transformer only in the browser.

Serves the repo root so the page can fetch both `web/*` assets and
`data/brain_reference/*` fixtures, opens `phase2.html`, runs the in-page
`runPhase2(backend)` entry, and reports per-cell cosine + max diff.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORT = 8767


def start_server(port: int, directory: str):
    handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=directory, **kw)  # noqa
    httpd = http.server.HTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--backends", nargs="+", default=["webgpu"])
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--timeout", type=int, default=180_000,
                    help="Per-call timeout in ms (default 180s)")
    args = ap.parse_args()

    # Serve repo root so the page can reach /web/* and /data/brain_reference/*.
    os.chdir(REPO_ROOT)
    httpd = start_server(args.port, str(REPO_ROOT))
    url = f"http://127.0.0.1:{args.port}/web/phase2.html"
    print(f"serving {REPO_ROOT} on :{args.port}")
    print(f"url: {url}")

    results: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=not args.headed,
            args=[
                "--enable-features=Vulkan,WebGPU",
                "--enable-unsafe-webgpu",
                "--use-angle=metal",
            ],
        )
        page = browser.new_page()
        page.set_default_timeout(args.timeout)
        page.on("console", lambda m: print(f"  [browser {m.type}] {m.text}"))
        page.on("pageerror", lambda e: print(f"  [browser error] {e}"))
        page.goto(url, wait_until="networkidle")
        print("page loaded\n")

        for backend in args.backends:
            print(f"=== backend: {backend} ===")
            try:
                page.evaluate(f"runPhase2('{backend}')")
                r = page.evaluate("window.phase2Result")
                results.append(r)
                print(f"  min_cosine: {r['min_cosine']:.6f}  "
                      f"max_diff: {r['max_diff']:.2e}  "
                      f"mean_ms: {r['mean_ms']:.1f}")
                status = "PASS" if r["min_cosine"] > 0.999 else "FAIL"
                print(f"  {status}\n")
            except Exception as e:
                print(f"  FAILED: {e}\n")
                results.append({"backend": backend, "error": str(e)})

        browser.close()
    httpd.shutdown()

    print(json.dumps(results, indent=2, default=float))

    # Exit nonzero if any backend failed the cos > 0.999 target.
    for r in results:
        if "error" in r or r.get("min_cosine", 0) <= 0.999:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
