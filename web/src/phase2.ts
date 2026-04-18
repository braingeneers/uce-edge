/**
 * Phase 2: Browser runs the transformer only.
 *
 * Python did: stages 1-7 (up through src_embeddings + attention_mask).
 * JS does: load pre-computed src + mask, run BrainCore ONNX on WebGPU,
 *          compare cell embeddings against the reference.
 *
 * Expected: cosine > 0.999 vs ref_cell_embedding.bin (we've seen 1.000000
 * on synthetic inputs previously, and this is the same graph with real
 * activation distributions).
 */
import { fetchJSON, loadFloat32, productOf, FixtureManifest } from "./lib/fixtures.js";
import { cosine, maxAbsDiff } from "./lib/metrics.js";
import { createBrainSession, runBrain, Backend } from "./lib/onnx.js";

const MODEL_URL = "/web/brain_8l_fp32.onnx";
const FIXTURES_BASE = "/data/brain_reference";

const logEl = document.getElementById("log")!;

function log(msg: string): void {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

function clearLog(): void {
  logEl.textContent = "";
}

async function run(backend: Backend): Promise<void> {
  clearLog();
  log(`phase 2: transformer-only, backend=${backend}`);

  log("loading manifest");
  const manifest = await fetchJSON<FixtureManifest>(`${FIXTURES_BASE}/manifest.json`);
  const { n_cells, pad_length: L, embedding_dim: D, d_model } = manifest;
  log(`  n_cells=${n_cells}, seq_len=${L}, emb_dim=${D}, d_model=${d_model}`);

  log("loading fixtures");
  const srcShape = [n_cells, L, D];
  const maskShape = [n_cells, L];
  const cellShape = [n_cells, d_model];

  const [src, mask, refCell] = await Promise.all([
    loadFloat32(`${FIXTURES_BASE}/ref_src_embeddings.bin`, productOf(srcShape)),
    loadFloat32(`${FIXTURES_BASE}/ref_attention_mask.bin`, productOf(maskShape)),
    loadFloat32(`${FIXTURES_BASE}/ref_cell_embedding.bin`, productOf(cellShape)),
  ]);
  log(`  src=${src.byteLength / 1e6} MB, mask=${mask.byteLength} B, ref_cell=${refCell.byteLength} B`);

  log(`creating ONNX session (${backend})`);
  const t0 = performance.now();
  const session = await createBrainSession(MODEL_URL, backend);
  log(`  session created in ${(performance.now() - t0).toFixed(0)} ms`);

  // Run cell-by-cell. BrainCore supports batching, but running one at a time
  // matches how the eventual browser demo will behave and keeps peak GPU
  // memory bounded.
  const results: { cell: number; cos: number; maxDiff: number; ms: number }[] = [];
  const srcStride = L * D;
  const maskStride = L;
  const cellStride = d_model;

  // Warmup with cell 0 (discard timing).
  {
    const srcBatch = src.slice(0, srcStride);
    const maskBatch = mask.slice(0, maskStride);
    await runBrain(session, srcBatch, maskBatch, 1, L, D);
  }

  for (let i = 0; i < n_cells; i++) {
    const srcBatch = src.slice(i * srcStride, (i + 1) * srcStride);
    const maskBatch = mask.slice(i * maskStride, (i + 1) * maskStride);
    const refSlice = refCell.subarray(i * cellStride, (i + 1) * cellStride);

    const tStart = performance.now();
    const { cell } = await runBrain(session, srcBatch, maskBatch, 1, L, D);
    const ms = performance.now() - tStart;

    const cos = cosine(cell, refSlice);
    const md = maxAbsDiff(cell, refSlice);
    results.push({ cell: i, cos, maxDiff: md, ms });
    log(`  cell ${i}: cos=${cos.toFixed(6)}  maxdiff=${md.toExponential(2)}  ${ms.toFixed(1)} ms`);
  }

  const minCos = Math.min(...results.map((r) => r.cos));
  const maxDiff = Math.max(...results.map((r) => r.maxDiff));
  const meanMs = results.reduce((a, r) => a + r.ms, 0) / results.length;

  log("\n=== summary ===");
  log(`  min cosine:   ${minCos.toFixed(6)}  (target > 0.999)`);
  log(`  max |diff|:   ${maxDiff.toExponential(2)}`);
  log(`  mean per-cell: ${meanMs.toFixed(1)} ms`);
  log(`  ${minCos > 0.999 ? "PASS" : "FAIL"}`);

  // Expose for Playwright.
  (window as unknown as { phase2Result: unknown }).phase2Result = {
    backend,
    n_cells,
    min_cosine: minCos,
    max_diff: maxDiff,
    mean_ms: meanMs,
    per_cell: results,
  };
}

function setup(): void {
  const runBtn = document.getElementById("run") as HTMLButtonElement;
  const backendSel = document.getElementById("backend") as HTMLSelectElement;
  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    try {
      await run(backendSel.value as Backend);
    } catch (e) {
      log(`\nERROR: ${(e as Error).message}`);
      log((e as Error).stack ?? "");
    } finally {
      runBtn.disabled = false;
    }
  });
}

setup();

// Programmatic entry for Playwright.
(window as unknown as { runPhase2: (b: Backend) => Promise<void> }).runPhase2 = run;
