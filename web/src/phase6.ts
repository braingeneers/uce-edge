/**
 * Phase 6: Full browser pipeline from raw counts + dynamic seq_len (Phase 7
 * optimization folded in here since it's trivial once the valid-length is
 * known at sentence-build time).
 *
 * JS does stages 1-7: log1p + sum-to-1 normalize, weighted sampling, chrom
 * shuffle, sentence build, gather, transformer. Python provides only the raw
 * counts (as if caller loaded an h5ad into memory) and the reference cell
 * embedding.
 *
 * Dynamic seq_len: padding is always at the sentence tail, so we trim src +
 * mask to the valid prefix length before calling the ONNX session. The
 * exported model has dynamic seq_len axes so no re-export is needed. At
 * ~1072 valid tokens vs 2048 pad_length this cuts attention work ~3.65x.
 *
 * Validation:
 *   - weights: element-wise vs ref_normalized_weights.bin (should be ~1e-7)
 *   - cell embedding: cos vs ref_cell_embedding.bin at Phase 5 noise floor
 *     (min > 0.88, mean > 0.90, hist pearson > 0.95)
 */
import { cachedFetchBytes } from "./lib/cache.js";
import { fetchJSON, loadFloat32, FixtureManifest } from "./lib/fixtures.js";
import { gather } from "./lib/gather.js";
import { cosine, maxAbsDiff } from "./lib/metrics.js";
import { normalizeCounts } from "./lib/normalize.js";
import { createBrainSession, runBrain, Backend } from "./lib/onnx.js";
import { RNG } from "./lib/rng.js";
import { weightedSample } from "./lib/sampler.js";
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
  specials: SentenceTokens["specials"];
  chromosome_token_map: Record<string, number>;
}

interface GeneSymbolsFixture {
  gene_protein_ids_new: number[];
  gene_chroms: number[];
  gene_starts: number[];
}

function pearson(a: Float64Array, b: Float64Array): number {
  if (a.length !== b.length) throw new Error("length mismatch");
  let ma = 0, mb = 0;
  for (let i = 0; i < a.length; i++) { ma += a[i]; mb += b[i]; }
  ma /= a.length; mb /= b.length;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < a.length; i++) {
    const xa = a[i] - ma;
    const xb = b[i] - mb;
    num += xa * xb;
    da += xa * xa;
    db += xb * xb;
  }
  return num / (Math.sqrt(da * db) + 1e-12);
}

async function run(backend: Backend): Promise<void> {
  clearLog();
  log(`phase 6: raw counts -> normalize -> sample -> sentence -> transformer, backend=${backend}`);

  log("loading gene dict");
  const geneDict = await fetchJSON<GeneDict>(GENE_DICT_URL);
  const tableRows = geneDict.num_rows;
  const embDim = geneDict.embedding_dim;

  const chromosomeTokenMap: Record<number, number> = {};
  for (const [k, v] of Object.entries(geneDict.chromosome_token_map)) {
    chromosomeTokenMap[parseInt(k, 10)] = v;
  }
  const tokens: SentenceTokens = { specials: geneDict.specials, chromosomeTokenMap };

  log("loading manifest + fixtures");
  const manifest = await fetchJSON<FixtureManifest>(`${FIXTURES_BASE}/manifest.json`);
  const { n_cells, n_aligned_genes: G, pad_length: L, embedding_dim: D, d_model, sample_size: S } = manifest;
  if (D !== embDim) throw new Error(`manifest emb_dim ${D} != gene dict ${embDim}`);

  const symbols = await fetchJSON<GeneSymbolsFixture>(`${FIXTURES_BASE}/ref_gene_symbols.json`);
  const table: GeneTable = {
    proteinIdsNew: new Int32Array(symbols.gene_protein_ids_new),
    chroms: new Int32Array(symbols.gene_chroms),
    starts: new Int32Array(symbols.gene_starts),
  };
  if (table.proteinIdsNew.length !== G) {
    throw new Error(`symbols ${table.proteinIdsNew.length} != manifest G ${G}`);
  }

  const [rawCountsAll, refWeightsAll, refCell] = await Promise.all([
    loadFloat32(`${FIXTURES_BASE}/ref_raw_counts.bin`, n_cells * G),
    loadFloat32(`${FIXTURES_BASE}/ref_normalized_weights.bin`, n_cells * G),
    loadFloat32(`${FIXTURES_BASE}/ref_cell_embedding.bin`, n_cells * d_model),
  ]);
  log(`  n_cells=${n_cells}, n_genes=${G}, sample_size=${S}, pad_length=${L}`);

  log(`fetching protein embedding table (${(tableRows * embDim * 4 / 1e6).toFixed(0)} MB)`);
  const tableResult = await cachedFetchBytes(EMBEDDING_URL);
  const embeddingTable = new Float32Array(tableResult.buffer);
  log(`  ${tableResult.fromCache ? "cache HIT" : "cache MISS"}  ${tableResult.ms.toFixed(0)} ms`);

  // ----------------------------------------------------------------------
  // Stage 2 in JS: log1p + sum-to-1 normalize from raw counts. Validate
  // element-wise vs Python's normalized weights (deterministic path).
  // ----------------------------------------------------------------------
  log("normalizing (log1p + sum-to-1)");
  const tNorm0 = performance.now();
  const weightsAll = new Float32Array(n_cells * G);
  for (let c = 0; c < n_cells; c++) {
    const raw = rawCountsAll.subarray(c * G, (c + 1) * G);
    const w = normalizeCounts(raw);
    weightsAll.set(w, c * G);
  }
  const normMs = performance.now() - tNorm0;
  const weightsDiff = maxAbsDiff(weightsAll, refWeightsAll);
  log(`  normalize: ${normMs.toFixed(1)} ms total`);
  log(`  weights max |diff|: ${weightsDiff.toExponential(2)}  (expect ~1e-7)`);

  // ----------------------------------------------------------------------
  // Stages 3-7 per cell (same as Phase 5).
  // ----------------------------------------------------------------------
  log("sampling + building sentences");
  const tBuild0 = performance.now();
  const tokenIdsAll = new Int32Array(n_cells * L);
  const attnMaskAll = new Float32Array(n_cells * L);

  for (let c = 0; c < n_cells; c++) {
    const rng = new RNG(c);
    const w = weightsAll.subarray(c * G, (c + 1) * G);
    const sampleIndices = weightedSample(w, S, rng);

    const chromSet = new Set<number>();
    for (let i = 0; i < sampleIndices.length; i++) {
      chromSet.add(table.chroms[sampleIndices[i]]);
    }
    const chromArr = Array.from(chromSet).sort((a, b) => a - b);
    rng.shuffle(chromArr);
    const chromOrder = Int32Array.from(chromArr);

    const { tokenIds, attentionMask } = buildSentence(sampleIndices, chromOrder, table, tokens, L);
    tokenIdsAll.set(tokenIds, c * L);
    attnMaskAll.set(attentionMask, c * L);
  }
  const buildMs = performance.now() - tBuild0;
  log(`  build: ${buildMs.toFixed(0)} ms total`);

  // Compute per-cell valid length from the attention mask. Padding is always
  // at the tail, so the valid prefix length == sum(mask). Gather happens
  // inside the per-cell loop below to avoid a 4GB one-shot allocation at
  // n_cells=100.
  const validLens = new Int32Array(n_cells);
  for (let c = 0; c < n_cells; c++) {
    let v = 0;
    const off = c * L;
    for (let j = 0; j < L; j++) v += attnMaskAll[off + j];
    validLens[c] = v;
  }
  let vMin = validLens[0], vMax = validLens[0], vSum = 0;
  for (let c = 0; c < n_cells; c++) {
    const v = validLens[c];
    if (v < vMin) vMin = v;
    if (v > vMax) vMax = v;
    vSum += v;
  }
  const vMean = vSum / n_cells;
  log(`  valid seq_len: min=${vMin}  max=${vMax}  mean=${vMean.toFixed(1)}  (pad_length=${L}, utilization=${(vMean/L*100).toFixed(1)}%)`);

  log(`creating ONNX session (${backend})`);
  const session = await createBrainSession(MODEL_URL, backend);

  const results: { cell: number; cos: number; maxDiff: number; ms: number; seqLen: number }[] = [];
  const maskStride = L;
  const cellStride = d_model;

  // Warmup at the expected valid length.
  const warmupLen = validLens[0];
  const warmupIds = tokenIdsAll.subarray(0, warmupLen);
  const warmupSrc = gather(embeddingTable, tableRows, embDim, warmupIds, 1, warmupLen);
  await runBrain(
    session,
    warmupSrc,
    attnMaskAll.subarray(0, warmupLen),
    1, warmupLen, D
  );

  const verbose = n_cells <= 10;
  for (let i = 0; i < n_cells; i++) {
    const validLen = validLens[i];
    const idsSlice = tokenIdsAll.subarray(i * L, i * L + validLen);
    const srcBatch = gather(embeddingTable, tableRows, embDim, idsSlice, 1, validLen);
    const maskBatch = attnMaskAll.slice(i * maskStride, i * maskStride + validLen);
    const refSlice = refCell.subarray(i * cellStride, (i + 1) * cellStride);

    const tStart = performance.now();
    const { cell } = await runBrain(session, srcBatch, maskBatch, 1, validLen, D);
    const ms = performance.now() - tStart;

    const cos = cosine(cell, refSlice);
    const md = maxAbsDiff(cell, refSlice);
    results.push({ cell: i, cos, maxDiff: md, ms, seqLen: validLen });
    if (verbose || i % 10 === 0 || i === n_cells - 1) {
      log(`  cell ${i}: seq_len=${validLen}  cos=${cos.toFixed(6)}  maxdiff=${md.toExponential(2)}  ${ms.toFixed(1)} ms`);
    }
  }

  const minCos = Math.min(...results.map((r) => r.cos));
  const meanCos = results.reduce((a, r) => a + r.cos, 0) / results.length;
  const meanMs = results.reduce((a, r) => a + r.ms, 0) / results.length;

  // Histogram check (cell 0) — same as Phase 5, proves the full pipeline
  // with JS-normalized weights still produces the right sampling distribution.
  log("\nhistogram check (cell 0, JS-normalized weights)");
  const trials = 1000;
  const hist = new Float64Array(G);
  const w0 = weightsAll.subarray(0, G);
  const histRng = new RNG(1_000_003);
  for (let t = 0; t < trials; t++) {
    const s = weightedSample(w0, S, histRng);
    for (let i = 0; i < s.length; i++) hist[s[i]]++;
  }
  const expected = new Float64Array(G);
  for (let i = 0; i < G; i++) expected[i] = w0[i] * trials * S;
  const r = pearson(hist, expected);
  log(`  trials=${trials}  pearson(obs, expected)=${r.toFixed(4)}  (target > 0.95)`);

  log("\n=== summary ===");
  log(`  weights diff: ${weightsDiff.toExponential(2)}  (expect ~1e-7)`);
  log(`  min cosine:   ${minCos.toFixed(6)}       (target > 0.88)`);
  log(`  mean cosine:  ${meanCos.toFixed(6)}      (target > 0.90)`);
  log(`  hist pearson: ${r.toFixed(4)}           (target > 0.95)`);
  log(`  normalize:    ${normMs.toFixed(1)} ms total`);
  log(`  mean per-cell transformer: ${meanMs.toFixed(1)} ms  (seq_len mean=${vMean.toFixed(1)}, pad=${L})`);
  // Phase 6 floor == Phase 5 floor (normalize is deterministic; drift source
  // is still the JS vs numpy sampler). Weights diff should be near Float32 eps.
  const pass =
    weightsDiff < 1e-5 &&
    minCos > 0.88 &&
    meanCos > 0.90 &&
    r > 0.95;
  log(`  ${pass ? "PASS" : "FAIL"}`);

  (window as unknown as { phase6Result: unknown }).phase6Result = {
    backend,
    n_cells,
    weights_max_diff: weightsDiff,
    min_cosine: minCos,
    mean_cosine: meanCos,
    hist_pearson: r,
    hist_trials: trials,
    normalize_ms: normMs,
    build_ms: buildMs,
    mean_ms: meanMs,
    seq_len_pad: L,
    seq_len_min: vMin,
    seq_len_max: vMax,
    seq_len_mean: vMean,
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

(window as unknown as { runPhase6: (b: Backend) => Promise<void> }).runPhase6 = run;
