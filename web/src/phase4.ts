/**
 * Phase 4: Browser does chromosome ordering + CLS/CHROM/PAD inserts + gather
 * + transformer.
 *
 * Python did: stages 1-3 only (weighted sampling). JS takes the sampled gene
 * indices + shuffled chromosome order as inputs and builds the rest.
 *
 * Validation (element-wise — no RNG on JS side yet):
 *   - token IDs match ref_ordered_token_ids_new.bin
 *   - cell embedding matches ref_cell_embedding.bin (cos > 0.999)
 */
import { cachedFetchBytes } from "./lib/cache.js";
import { fetchJSON, loadFloat32, loadInt32, FixtureManifest } from "./lib/fixtures.js";
import { gather } from "./lib/gather.js";
import { cosine, maxAbsDiff } from "./lib/metrics.js";
import { createBrainSession, runBrain, Backend } from "./lib/onnx.js";
import { buildSentence, GeneTable, SentenceTokens } from "./lib/sentence.js";

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
  specials: SentenceTokens["specials"];
  chromosome_token_map: Record<string, number>;
}

interface GeneSymbolsFixture {
  aligned_gene_names: string[];
  gene_protein_ids_old: number[];
  gene_protein_ids_new: number[];
  gene_chroms: number[];
  gene_starts: number[];
  pad_length: number;
  sample_size: number;
}

async function run(backend: Backend): Promise<void> {
  clearLog();
  log(`phase 4: chrom-order + gather + transformer, backend=${backend}`);

  log("loading gene dict");
  const geneDict = await fetchJSON<GeneDict>(GENE_DICT_URL);
  const tableRows = geneDict.num_rows;
  const embDim = geneDict.embedding_dim;
  log(`  table=${tableRows}x${embDim}  layer_norm_applied=${geneDict.layer_norm_applied}`);

  // Chromosome map JSON keys are strings — normalize to number keys.
  const chromosomeTokenMap: Record<number, number> = {};
  for (const [k, v] of Object.entries(geneDict.chromosome_token_map)) {
    chromosomeTokenMap[parseInt(k, 10)] = v;
  }

  const tokens: SentenceTokens = {
    specials: geneDict.specials,
    chromosomeTokenMap,
  };

  log("loading manifest + fixtures");
  const manifest = await fetchJSON<FixtureManifest>(`${FIXTURES_BASE}/manifest.json`);
  const { n_cells, pad_length: L, embedding_dim: D, d_model, sample_size: S } = manifest;
  if (D !== embDim) throw new Error(`manifest emb_dim ${D} != gene dict ${embDim}`);

  const symbols = await fetchJSON<GeneSymbolsFixture>(`${FIXTURES_BASE}/ref_gene_symbols.json`);
  const table: GeneTable = {
    proteinIdsNew: new Int32Array(symbols.gene_protein_ids_new),
    chroms: new Int32Array(symbols.gene_chroms),
    starts: new Int32Array(symbols.gene_starts),
  };
  log(`  aligned genes: ${table.proteinIdsNew.length}`);

  const maxChroms = manifest.files["ref_chrom_order.bin"].shape[1];

  const [sampleIndices, chromOrders, refTokenIds, refAttn, refCell] = await Promise.all([
    loadInt32(`${FIXTURES_BASE}/ref_sample_indices.bin`, n_cells * S),
    loadInt32(`${FIXTURES_BASE}/ref_chrom_order.bin`, n_cells * maxChroms),
    loadInt32(`${FIXTURES_BASE}/ref_ordered_token_ids_new.bin`, n_cells * L),
    loadFloat32(`${FIXTURES_BASE}/ref_attention_mask.bin`, n_cells * L),
    loadFloat32(`${FIXTURES_BASE}/ref_cell_embedding.bin`, n_cells * d_model),
  ]);
  log(`  n_cells=${n_cells}, sample_size=${S}, max_chroms=${maxChroms}`);

  log(`fetching protein embedding table (${(tableRows * embDim * 4 / 1e6).toFixed(0)} MB)`);
  const tableResult = await cachedFetchBytes(EMBEDDING_URL);
  const embeddingTable = new Float32Array(tableResult.buffer);
  log(`  ${tableResult.fromCache ? "cache HIT" : "cache MISS"}  ${tableResult.ms.toFixed(0)} ms`);

  // ----------------------------------------------------------------------
  // Stages 4-6 in JS: build sentence + mask per cell, then gather.
  // ----------------------------------------------------------------------
  log("building sentences");
  const tBuild0 = performance.now();
  const tokenIdsAll = new Int32Array(n_cells * L);
  const attnMaskAll = new Float32Array(n_cells * L);

  for (let c = 0; c < n_cells; c++) {
    const sampleSlice = sampleIndices.subarray(c * S, (c + 1) * S);
    const chromSlice = chromOrders.subarray(c * maxChroms, (c + 1) * maxChroms);
    const { tokenIds, attentionMask } = buildSentence(sampleSlice, chromSlice, table, tokens, L);
    tokenIdsAll.set(tokenIds, c * L);
    attnMaskAll.set(attentionMask, c * L);
  }
  const buildMs = performance.now() - tBuild0;
  log(`  build: ${buildMs.toFixed(0)} ms total`);

  // Element-wise match: token IDs and attention mask.
  const tokenDiff = maxAbsDiff(
    new Float32Array(tokenIdsAll.buffer, tokenIdsAll.byteOffset, tokenIdsAll.length),
    new Float32Array(refTokenIds.buffer, refTokenIds.byteOffset, refTokenIds.length)
  );
  const maskDiff = maxAbsDiff(attnMaskAll, refAttn);
  log(`  token_ids max |diff|: ${tokenDiff} (expect 0)`);
  log(`  attn_mask max |diff|: ${maskDiff} (expect 0)`);

  // Find first divergence for diagnostics if any.
  if (tokenDiff !== 0) {
    for (let i = 0; i < tokenIdsAll.length; i++) {
      if (tokenIdsAll[i] !== refTokenIds[i]) {
        const c = Math.floor(i / L);
        const p = i % L;
        log(`  first diff: cell ${c} pos ${p}  js=${tokenIdsAll[i]} ref=${refTokenIds[i]}`);
        break;
      }
    }
  }

  // ----------------------------------------------------------------------
  // Gather + transformer (reuse Phase 3 path).
  // ----------------------------------------------------------------------
  log("gathering embeddings");
  const src = gather(embeddingTable, tableRows, embDim, tokenIdsAll, n_cells, L);

  log(`creating ONNX session (${backend})`);
  const session = await createBrainSession(MODEL_URL, backend);

  const results: { cell: number; cos: number; maxDiff: number; ms: number }[] = [];
  const srcStride = L * D;
  const maskStride = L;
  const cellStride = d_model;

  // Warmup.
  await runBrain(
    session,
    src.subarray(0, srcStride),
    attnMaskAll.subarray(0, maskStride),
    1, L, D
  );

  for (let i = 0; i < n_cells; i++) {
    const srcBatch = src.slice(i * srcStride, (i + 1) * srcStride);
    const maskBatch = attnMaskAll.slice(i * maskStride, (i + 1) * maskStride);
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
  log(`  token_ids diff: ${tokenDiff} (expect 0)`);
  log(`  attn_mask diff: ${maskDiff} (expect 0)`);
  log(`  min cosine:     ${minCos.toFixed(6)} (target > 0.999)`);
  log(`  mean per-cell:  ${meanMs.toFixed(1)} ms`);
  const pass = tokenDiff === 0 && maskDiff === 0 && minCos > 0.999;
  log(`  ${pass ? "PASS" : "FAIL"}`);

  (window as unknown as { phase4Result: unknown }).phase4Result = {
    backend,
    n_cells,
    token_diff: tokenDiff,
    mask_diff: maskDiff,
    min_cosine: minCos,
    mean_ms: meanMs,
    build_ms: buildMs,
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

(window as unknown as { runPhase4: (b: Backend) => Promise<void> }).runPhase4 = run;
