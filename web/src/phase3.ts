/**
 * Phase 3: Browser does the protein embedding gather + transformer.
 *
 * Python did: stages 1-5 (up through renumbered token IDs).
 * JS does: cache-fetch the embedding table, gather to (B, L, 5120),
 *          run through ONNX, compare against the reference.
 *
 * Validation:
 *   - reconstructed src matches ref_src_embeddings.bin element-wise (0 diff)
 *   - cell embedding matches ref_cell_embedding.bin (cos > 0.999)
 */
import { cachedFetchBytes } from "./lib/cache.js";
import { fetchJSON, loadFloat32, loadInt32, productOf, FixtureManifest } from "./lib/fixtures.js";
import { gather } from "./lib/gather.js";
import { cosine, maxAbsDiff } from "./lib/metrics.js";
import { createBrainSession, runBrain, Backend } from "./lib/onnx.js";

const MODEL_URL = "/web/brain_8l_fp32.onnx";
const EMBEDDING_URL = "/web/human_protein_embeddings.bin";
const GENE_DICT_URL = "/web/human_gene_dict.json";
const FIXTURES_BASE = "/data/brain_reference";

const logEl = document.getElementById("log")!;

function log(msg: string): void {
  logEl.textContent += msg + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

function clearLog(): void {
  logEl.textContent = "";
}

interface GeneDict {
  num_rows: number;
  embedding_dim: number;
  layer_norm_applied: boolean;
}

async function run(backend: Backend): Promise<void> {
  clearLog();
  log(`phase 3: gather + transformer, backend=${backend}`);

  log("loading gene dict");
  const geneDict = await fetchJSON<GeneDict>(GENE_DICT_URL);
  const tableRows = geneDict.num_rows;
  const embDim = geneDict.embedding_dim;
  log(`  table=${tableRows}x${embDim}  layer_norm_applied=${geneDict.layer_norm_applied}`);

  log("loading manifest");
  const manifest = await fetchJSON<FixtureManifest>(`${FIXTURES_BASE}/manifest.json`);
  const { n_cells, pad_length: L, embedding_dim: D, d_model } = manifest;
  if (D !== embDim) throw new Error(`manifest emb_dim ${D} != gene dict ${embDim}`);
  log(`  n_cells=${n_cells}, seq_len=${L}, emb_dim=${D}, d_model=${d_model}`);

  log(`fetching protein embedding table (${(tableRows * embDim * 4 / 1e6).toFixed(0)} MB)`);
  const tableResult = await cachedFetchBytes(EMBEDDING_URL);
  const table = new Float32Array(tableResult.buffer);
  if (table.length !== tableRows * embDim) {
    throw new Error(`table length ${table.length} != ${tableRows * embDim}`);
  }
  log(`  ${tableResult.fromCache ? "cache HIT" : "cache MISS"}  ${tableResult.ms.toFixed(0)} ms`);

  log("loading fixtures");
  const [tokenIds, mask, refSrc, refCell] = await Promise.all([
    loadInt32(`${FIXTURES_BASE}/ref_ordered_token_ids_new.bin`, n_cells * L),
    loadFloat32(`${FIXTURES_BASE}/ref_attention_mask.bin`, n_cells * L),
    loadFloat32(`${FIXTURES_BASE}/ref_src_embeddings.bin`, productOf([n_cells, L, D])),
    loadFloat32(`${FIXTURES_BASE}/ref_cell_embedding.bin`, n_cells * d_model),
  ]);
  log(`  token_ids=${tokenIds.byteLength / 1e3} KB, ref_src=${(refSrc.byteLength / 1e6).toFixed(0)} MB`);

  // ----------------------------------------------------------------------
  // Stage 7: gather. Run on full batch, validate element-wise vs ref_src.
  // ----------------------------------------------------------------------
  log("gathering embeddings");
  const tGather0 = performance.now();
  const src = gather(table, tableRows, embDim, tokenIds, n_cells, L);
  const gatherMs = performance.now() - tGather0;
  log(`  gather: ${gatherMs.toFixed(0)} ms`);

  const gatherMaxDiff = maxAbsDiff(src, refSrc);
  log(`  gather max |diff| vs ref_src: ${gatherMaxDiff.toExponential(2)} (expect 0)`);
  if (gatherMaxDiff !== 0) {
    log("  WARNING: gather does not match reference element-wise");
  }

  // ----------------------------------------------------------------------
  // Transformer: one cell at a time.
  // ----------------------------------------------------------------------
  log(`creating ONNX session (${backend})`);
  const tSess0 = performance.now();
  const session = await createBrainSession(MODEL_URL, backend);
  log(`  session created in ${(performance.now() - tSess0).toFixed(0)} ms`);

  const results: { cell: number; cos: number; maxDiff: number; ms: number }[] = [];
  const srcStride = L * D;
  const maskStride = L;
  const cellStride = d_model;

  // Warmup
  await runBrain(
    session,
    src.subarray(0, srcStride),
    mask.subarray(0, maskStride),
    1, L, D
  );

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
  const meanMs = results.reduce((a, r) => a + r.ms, 0) / results.length;

  log("\n=== summary ===");
  log(`  gather max |diff|: ${gatherMaxDiff.toExponential(2)} (expect 0)`);
  log(`  min cosine:        ${minCos.toFixed(6)} (target > 0.999)`);
  log(`  mean per-cell:     ${meanMs.toFixed(1)} ms`);
  const pass = gatherMaxDiff === 0 && minCos > 0.999;
  log(`  ${pass ? "PASS" : "FAIL"}`);

  (window as unknown as { phase3Result: unknown }).phase3Result = {
    backend,
    n_cells,
    table_from_cache: tableResult.fromCache,
    table_ms: tableResult.ms,
    gather_ms: gatherMs,
    gather_max_diff: gatherMaxDiff,
    min_cosine: minCos,
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

  const clearBtn = document.getElementById("clear-cache") as HTMLButtonElement | null;
  if (clearBtn) {
    clearBtn.addEventListener("click", async () => {
      const names = await caches.keys();
      for (const n of names) await caches.delete(n);
      log(`cleared caches: ${names.join(", ") || "(none)"}`);
    });
  }
}

setup();

(window as unknown as { runPhase3: (b: Backend) => Promise<void> }).runPhase3 = run;
