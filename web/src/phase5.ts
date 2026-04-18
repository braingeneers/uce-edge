/**
 * Phase 5: Browser does weighted sampling too. Python provides only
 * normalized weights; JS does stages 3-7.
 *
 * This is the first phase where bit-identity fails (JS Math.random / Mulberry32
 * != numpy PCG64). Validation shifts to:
 *   - per-cell cosine between JS and Python cell embeddings. Targets are set
 *     against the intrinsic Python-vs-Python seed-noise floor: on this h5ad
 *     cell 0 has sparse counts and even Python-vs-Python at different seeds
 *     only hits ~0.89–0.94. So > 0.88 individual / > 0.90 mean is realistic.
 *   - distributional check: for one cell, sample many times and compare the
 *     gene-frequency histogram to Python's expected histogram (weights *
 *     n_samples). Pearson > 0.95 — this is the authoritative sampler check.
 */
import { cachedFetchBytes } from "./lib/cache.js";
import { fetchJSON, loadFloat32, FixtureManifest } from "./lib/fixtures.js";
import { gather } from "./lib/gather.js";
import { cosine, maxAbsDiff } from "./lib/metrics.js";
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
  log(`phase 5: weighted sampling + chrom-order + gather + transformer, backend=${backend}`);

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

  const [weightsAll, refCell] = await Promise.all([
    loadFloat32(`${FIXTURES_BASE}/ref_normalized_weights.bin`, n_cells * G),
    loadFloat32(`${FIXTURES_BASE}/ref_cell_embedding.bin`, n_cells * d_model),
  ]);
  log(`  n_cells=${n_cells}, n_genes=${G}, sample_size=${S}, pad_length=${L}`);

  log(`fetching protein embedding table (${(tableRows * embDim * 4 / 1e6).toFixed(0)} MB)`);
  const tableResult = await cachedFetchBytes(EMBEDDING_URL);
  const embeddingTable = new Float32Array(tableResult.buffer);
  log(`  ${tableResult.fromCache ? "cache HIT" : "cache MISS"}  ${tableResult.ms.toFixed(0)} ms`);

  // ----------------------------------------------------------------------
  // Per-cell: sample, shuffle chroms, build sentence.
  // ----------------------------------------------------------------------
  log("sampling + building sentences");
  const tBuild0 = performance.now();
  const tokenIdsAll = new Int32Array(n_cells * L);
  const attnMaskAll = new Float32Array(n_cells * L);

  for (let c = 0; c < n_cells; c++) {
    // One RNG per cell, seeded by cell index. This mirrors the Python
    // sampler's seed=idx convention (values differ; structure matches).
    const rng = new RNG(c);

    const w = weightsAll.subarray(c * G, (c + 1) * G);
    const sampleIndices = weightedSample(w, S, rng);

    // Unique chromosomes present in the sampled set, sorted ascending then
    // shuffled — matches the sampler's np.unique + rng.shuffle.
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

  // ----------------------------------------------------------------------
  // Gather + transformer.
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
  const meanCos = results.reduce((a, r) => a + r.cos, 0) / results.length;
  const meanMs = results.reduce((a, r) => a + r.ms, 0) / results.length;

  // ----------------------------------------------------------------------
  // Distributional check: for cell 0, resample many times, compare histogram
  // of sampled gene indices to the expected histogram (weights * trials).
  // ----------------------------------------------------------------------
  log("\nhistogram check (cell 0)");
  const trials = 1000;
  const hist = new Float64Array(G);
  const w0 = weightsAll.subarray(0, G);
  const histRng = new RNG(1_000_003); // distinct seed — doesn't reuse cell 0 draws
  for (let t = 0; t < trials; t++) {
    const s = weightedSample(w0, S, histRng);
    for (let i = 0; i < s.length; i++) hist[s[i]]++;
  }
  const expected = new Float64Array(G);
  for (let i = 0; i < G; i++) expected[i] = w0[i] * trials * S;
  const r = pearson(hist, expected);
  log(`  trials=${trials}  sample_size=${S}  pearson(obs, expected)=${r.toFixed(4)}  (target > 0.95)`);

  log("\n=== summary ===");
  log(`  min cosine:   ${minCos.toFixed(6)}  (target > 0.88)`);
  log(`  mean cosine:  ${meanCos.toFixed(6)} (target > 0.90)`);
  log(`  hist pearson: ${r.toFixed(4)}       (target > 0.95  — the authoritative sampler check)`);
  log(`  mean per-cell: ${meanMs.toFixed(1)} ms`);
  // Thresholds set against Python-vs-Python seed-noise floor on this h5ad
  // (cell 0 with sparse counts is noisiest at ~0.89). Histogram Pearson is
  // what actually proves the JS sampler matches the numpy distribution.
  const pass = minCos > 0.88 && meanCos > 0.90 && r > 0.95;
  log(`  ${pass ? "PASS" : "FAIL"}`);

  (window as unknown as { phase5Result: unknown }).phase5Result = {
    backend,
    n_cells,
    min_cosine: minCos,
    mean_cosine: meanCos,
    hist_pearson: r,
    hist_trials: trials,
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

(window as unknown as { runPhase5: (b: Backend) => Promise<void> }).runPhase5 = run;
