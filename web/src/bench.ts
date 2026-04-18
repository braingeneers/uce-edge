/**
 * Backend/option benchmark for UCE-brain transformer.
 *
 * Measures per-cell transformer latency for several ORT-Web configurations so
 * we can quantify headroom: WebGPU vs WASM, graph capture, preferred output
 * location, thread count.
 */
import * as ort from "onnxruntime-web";
import { cachedFetchBytes } from "./lib/cache.js";

ort.env.wasm.wasmPaths = "/web/dist/ort/";

const MODEL_URL = "/web/brain_8l_fp32.onnx";
const INT8_URL = "/web/brain_8l_int8.onnx";

const logEl = document.getElementById("log")!;
function log(msg: string): void {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}
function clearLog(): void { logEl.textContent = ""; }

interface Config {
  name: string;
  providers: (string | { name: string })[];
  modelUrl?: string;
  batchSize?: number;
  sessionOpts?: ort.InferenceSession.SessionOptions;
  wasmNumThreads?: number;
  wasmSimd?: boolean;
}

const L = 2048;
const D = 5120;

async function timeRun(
  session: ort.InferenceSession,
  src: Float32Array,
  mask: Float32Array,
  iters: number,
  batch: number
): Promise<number[]> {
  const times: number[] = [];
  // Warmup × 2
  for (let i = 0; i < 2; i++) {
    const srcT = new ort.Tensor("float32", src, [batch, L, D]);
    const maskT = new ort.Tensor("float32", mask, [batch, L]);
    await session.run({ src: srcT, mask: maskT });
  }
  for (let i = 0; i < iters; i++) {
    const srcT = new ort.Tensor("float32", src, [batch, L, D]);
    const maskT = new ort.Tensor("float32", mask, [batch, L]);
    const t0 = performance.now();
    const out = await session.run({ src: srcT, mask: maskT });
    void (out.cell_embedding.data as Float32Array)[0];
    times.push(performance.now() - t0);
  }
  return times;
}

function stats(ts: number[]): { mean: number; min: number; p50: number } {
  const sorted = [...ts].sort((a, b) => a - b);
  return {
    mean: ts.reduce((a, b) => a + b, 0) / ts.length,
    min: sorted[0],
    p50: sorted[Math.floor(sorted.length / 2)],
  };
}

async function runConfig(cfg: Config): Promise<void> {
  log(`\n=== ${cfg.name} ===`);
  if (cfg.wasmNumThreads !== undefined) ort.env.wasm.numThreads = cfg.wasmNumThreads;
  if (cfg.wasmSimd !== undefined) ort.env.wasm.simd = cfg.wasmSimd;
  const batch = cfg.batchSize ?? 1;
  const src = new Float32Array(batch * L * D);
  for (let i = 0; i < src.length; i++) src[i] = (Math.random() - 0.5) * 2;
  const mask = new Float32Array(batch * L).fill(1);
  try {
    const opts: ort.InferenceSession.SessionOptions = {
      executionProviders: cfg.providers as never,
      ...(cfg.sessionOpts ?? {}),
    };
    const session = await ort.InferenceSession.create(cfg.modelUrl ?? MODEL_URL, opts);
    const times = await timeRun(session, src, mask, 6, batch);
    const s = stats(times);
    const perCell = s.mean / batch;
    log(`  mean=${s.mean.toFixed(1)} ms (batch=${batch} -> ${perCell.toFixed(1)} ms/cell)  min=${s.min.toFixed(1)} ms  p50=${s.p50.toFixed(1)} ms`);
    (window as unknown as { benchResults: Record<string, unknown> }).benchResults[cfg.name] = {
      mean_ms: s.mean, min_ms: s.min, p50_ms: s.p50, per_cell_ms: perCell, batch, times,
    };
    await session.release();
  } catch (e) {
    log(`  FAILED: ${(e as Error).message}`);
    (window as unknown as { benchResults: Record<string, unknown> }).benchResults[cfg.name] = {
      error: (e as Error).message,
    };
  }
}

async function run(): Promise<void> {
  clearLog();
  (window as unknown as { benchResults: Record<string, unknown> }).benchResults = {};

  log(`hw concurrency: ${navigator.hardwareConcurrency}`);
  log(`webgpu: ${"gpu" in navigator ? "available" : "NOT available"}`);
  log(`SharedArrayBuffer: ${typeof SharedArrayBuffer !== "undefined" ? "available" : "NOT available (wasm threads will fall back)"}`);
  log(`crossOriginIsolated: ${typeof crossOriginIsolated !== "undefined" ? crossOriginIsolated : "unknown"}`);

  log("fetching models (cached after first load)");
  await cachedFetchBytes(MODEL_URL);
  await cachedFetchBytes(INT8_URL).catch(() => {});

  const configs: Config[] = [
    { name: "webgpu batch=1", providers: ["webgpu"] },
    { name: "webgpu batch=2", providers: ["webgpu"], batchSize: 2 },
    { name: "webgpu batch=4", providers: ["webgpu"], batchSize: 4 },
    { name: "webgpu int8 batch=1", providers: ["webgpu"], modelUrl: INT8_URL },
    { name: "wasm 1 thread (simd)", providers: ["wasm"], wasmNumThreads: 1, wasmSimd: true },
    { name: "wasm 4 threads (simd)", providers: ["wasm"], wasmNumThreads: 4, wasmSimd: true },
    { name: "wasm int8 1 thread", providers: ["wasm"], modelUrl: INT8_URL, wasmNumThreads: 1, wasmSimd: true },
    { name: "wasm int8 4 threads", providers: ["wasm"], modelUrl: INT8_URL, wasmNumThreads: 4, wasmSimd: true },
  ];

  for (const cfg of configs) {
    await runConfig(cfg);
  }

  log("\n=== done ===");
}

function setup(): void {
  const runBtn = document.getElementById("run") as HTMLButtonElement;
  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    try { await run(); } catch (e) {
      log(`\nERROR: ${(e as Error).message}`);
    } finally { runBtn.disabled = false; }
  });
}

setup();
(window as unknown as { runBench: () => Promise<void> }).runBench = run;
