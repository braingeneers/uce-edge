/**
 * Thin wrapper around onnxruntime-web for the UCE-brain session.
 *
 * We import the full bundle (.all.min.mjs) so WebGPU EP is available without
 * extra configuration. Bundling this inflates the JS output; acceptable for
 * the demo.
 */
import * as ort from "onnxruntime-web";

// ORT-web needs to fetch its WASM workers at runtime. The path is resolved
// relative to the page URL, so use an absolute URL path. The build step
// copies the workers into web/dist/ort/ (see Makefile: web-build-phase2).
ort.env.wasm.wasmPaths = "/web/dist/ort/";

export type Backend = "webgpu" | "wasm";

export async function createBrainSession(
  modelUrl: string,
  backend: Backend
): Promise<ort.InferenceSession> {
  const executionProviders = [{ name: backend }];
  return ort.InferenceSession.create(modelUrl, { executionProviders });
}

/**
 * Run BrainCore forward on a single batch.
 *
 * src:   (batch, seq_len, 5120) FP32 — post-LayerNorm embeddings
 * mask:  (batch, seq_len) FP32       — 1=valid, 0=pad
 *
 * Returns `cell_embedding` (batch, 512) and `gene_embeddings` (batch, seq_len, 512).
 */
export async function runBrain(
  session: ort.InferenceSession,
  src: Float32Array,
  mask: Float32Array,
  batchSize: number,
  seqLen: number,
  embDim: number
): Promise<{ cell: Float32Array; gene: Float32Array }> {
  const srcTensor = new ort.Tensor("float32", src, [batchSize, seqLen, embDim]);
  const maskTensor = new ort.Tensor("float32", mask, [batchSize, seqLen]);
  const out = await session.run({ src: srcTensor, mask: maskTensor });
  return {
    cell: out.cell_embedding.data as Float32Array,
    gene: out.gene_embeddings.data as Float32Array,
  };
}

export { ort };
