"""Playwright driver for the backend/options benchmark (bench.html)."""
from __future__ import annotations

import argparse
import http.server
import json
import os
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORT = 8772


def start_server(port: int, directory: str):
    # Send COOP/COEP so the browser enables SharedArrayBuffer (required for
    # WASM threading under ORT). Without these headers ORT silently falls back
    # to single-threaded regardless of numThreads setting.
    class Handler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
            super().end_headers()

    handler = lambda *a, **kw: Handler(*a, directory=directory, **kw)  # noqa
    httpd = http.server.HTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--timeout", type=int, default=600_000)
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    httpd = start_server(args.port, str(REPO_ROOT))
    url = f"http://127.0.0.1:{args.port}/web/bench.html"
    print(f"url: {url}\n")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=not args.headed,
            args=[
                "--enable-features=Vulkan,WebGPU,SharedArrayBuffer",
                "--enable-unsafe-webgpu",
                "--use-angle=metal",
            ],
        )
        context = browser.new_context(
            # SharedArrayBuffer (needed for wasm threading) requires cross-origin isolation.
            # SimpleHTTPRequestHandler doesn't send COOP/COEP headers, so wasm threads may fall back.
        )
        page = context.new_page()
        page.set_default_timeout(args.timeout)
        page.on("console", lambda m: print(f"  [browser {m.type}] {m.text}"))
        page.on("pageerror", lambda e: print(f"  [browser error] {e}"))
        page.goto(url, wait_until="networkidle")

        page.evaluate("runBench()")
        results = page.evaluate("window.benchResults")

        browser.close()
    httpd.shutdown()

    print("\n=== results ===")
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
