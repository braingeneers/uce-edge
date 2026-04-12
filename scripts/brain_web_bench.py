"""Automated browser benchmarks for UCE-brain ONNX via Playwright.

Launches a real Chrome with WebGPU, serves the web/ directory, and runs
all model × backend combinations programmatically.
"""
from __future__ import annotations

import argparse
import http.server
import json
import threading

from playwright.sync_api import sync_playwright

WEB_DIR = "web"
DEFAULT_PORT = 8766


def start_server(port: int):
    handler = http.server.SimpleHTTPRequestHandler
    httpd = http.server.HTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--models", nargs="+",
                    default=["brain_8l_fp32.onnx", "brain_8l_int8.onnx"])
    ap.add_argument("--backends", nargs="+", default=["webgpu", "wasm"])
    ap.add_argument("--headed", action="store_true",
                    help="Show the browser window")
    args = ap.parse_args()

    import os
    os.chdir(WEB_DIR)
    httpd = start_server(args.port)
    url = f"http://127.0.0.1:{args.port}/index.html"
    print(f"serving {WEB_DIR}/ on :{args.port}")

    results = []

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
        page.set_default_timeout(120_000)
        page.goto(url, wait_until="networkidle")
        print(f"page loaded: {url}\n")

        header = (f"{'model':<24} {'backend':<8} {'time_ms':>8} "
                  f"{'cell_maxdiff':>14} {'cosine':>10}")
        print(header)
        print("-" * len(header))

        for model in args.models:
            for backend in args.backends:
                try:
                    r = page.evaluate(
                        f"runBenchmark('{model}', '{backend}')"
                    )
                    results.append(r)
                    print(f"{r['model']:<24} {r['backend']:<8} "
                          f"{r['time_ms']:>8.1f} "
                          f"{r['cell_maxdiff']:>14.2e} "
                          f"{r['cosine']:>10.6f}")
                except Exception as e:
                    print(f"{model:<24} {backend:<8} FAILED: {e}")
                    results.append({
                        "model": model, "backend": backend,
                        "error": str(e),
                    })

        browser.close()
    httpd.shutdown()

    print(f"\n{json.dumps(results, indent=2)}")


if __name__ == "__main__":
    main()
