/**
 * Protein-embedding gather.
 *
 * Given an embedding table (numRows, embDim) FP32 and a batch of token IDs
 * (batchSize, seqLen), produce (batchSize, seqLen, embDim) FP32 where each
 * row is copied from the table by token ID.
 *
 * The implementation is a straight Float32Array copy per position — no math,
 * just memcpy. No randomness or numerical noise is introduced, so this should
 * match the Python reference `ref_src_embeddings.bin` element-wise.
 */

export function gather(
  table: Float32Array,
  tableRows: number,
  embDim: number,
  tokenIds: Int32Array,
  batchSize: number,
  seqLen: number
): Float32Array {
  if (table.length !== tableRows * embDim) {
    throw new Error(`table length ${table.length} != ${tableRows} * ${embDim}`);
  }
  if (tokenIds.length !== batchSize * seqLen) {
    throw new Error(`tokenIds length ${tokenIds.length} != ${batchSize} * ${seqLen}`);
  }
  const out = new Float32Array(batchSize * seqLen * embDim);
  for (let i = 0; i < tokenIds.length; i++) {
    const id = tokenIds[i];
    if (id < 0 || id >= tableRows) {
      throw new Error(`token id ${id} at position ${i} out of range [0, ${tableRows})`);
    }
    const srcOffset = id * embDim;
    const dstOffset = i * embDim;
    // Float32Array.set with a subarray is a zero-copy view + memcpy.
    out.set(table.subarray(srcOffset, srcOffset + embDim), dstOffset);
  }
  return out;
}
