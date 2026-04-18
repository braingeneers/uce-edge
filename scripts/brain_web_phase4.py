"""Playwright driver for Phase 4 — chromosome ordering in the browser."""
from __future__ import annotations

import argparse
import http.server
import json
import os
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORT = 8769


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
    ap.add_argument("--timeout", type=int, default=300_000)
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    httpd = start_server(args.port, str(REPO_ROOT))
    url = f"http://127.0.0.1:{args.port}/web/phase4.html"
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
                page.evaluate(f"runPhase4('{backend}')")
                r = page.evaluate("window.phase4Result")
                results.append(r)
                print(f"  token_diff: {r['token_diff']}  "
                      f"mask_diff: {r['mask_diff']}  "
                      f"min_cosine: {r['min_cosine']:.6f}  "
                      f"mean_ms: {r['mean_ms']:.1f}")
                ok = r["token_diff"] == 0 and r["mask_diff"] == 0 and r["min_cosine"] > 0.999
                print(f"  {'PASS' if ok else 'FAIL'}\n")
            except Exception as e:
                print(f"  FAILED: {e}\n")
                results.append({"backend": backend, "error": str(e)})

        browser.close()
    httpd.shutdown()

    print(json.dumps(results, indent=2, default=float))

    for r in results:
        if "error" in r:
            raise SystemExit(1)
        if r.get("token_diff", 1) != 0 or r.get("mask_diff", 1) != 0 or r.get("min_cosine", 0) <= 0.999:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
