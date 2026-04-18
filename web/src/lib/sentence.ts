/**
 * Build a UCE-brain cell sentence from a list of sampled gene indices.
 *
 * Mirrors UCE-brain/src/uce_brain/data/sampler.py, post-sampling:
 *   - CLS token
 *   - For each chromosome in `chromOrder` (already shuffled by Python):
 *       * chromosome boundary token (start marker)
 *       * genes on that chromosome, sorted ascending by genomic start position
 *       * CHROM_END marker
 *   - PAD tokens fill the rest up to padLength
 *
 * All IDs are in the "new" dense-index space (matching
 * web/human_protein_embeddings.bin), so the output plugs directly into gather.
 *
 * Returns { tokenIds (padLength int32), attentionMask (padLength float32) }.
 */

export interface SentenceTokens {
  specials: {
    pad_token_idx: number;
    cls_token_idx: number;
    chrom_token_right_idx: number;
  };
  /** Map from raw chromosome_id (int, e.g. 564..613) to new-space token id. */
  chromosomeTokenMap: Record<number, number>;
}

export interface GeneTable {
  /** new-space protein id for each aligned gene (length = numAlignedGenes). */
  proteinIdsNew: Int32Array;
  /** chromosome id for each aligned gene (int). */
  chroms: Int32Array;
  /** genomic start for each aligned gene. */
  starts: Int32Array;
}

export function buildSentence(
  sampleIndices: Int32Array,
  chromOrder: Int32Array, // -1 sentinel for unused slots
  table: GeneTable,
  tokens: SentenceTokens,
  padLength: number
): { tokenIds: Int32Array; attentionMask: Float32Array } {
  const tokenIds = new Int32Array(padLength);
  const attentionMask = new Float32Array(padLength);

  // Index 0: CLS
  tokenIds[0] = tokens.specials.cls_token_idx;
  attentionMask[0] = 1;
  let cursor = 1;

  // Group sampled gene indices by chromosome, so we only iterate each
  // chromosome's genes once when laying out the sentence.
  //
  // Build: chrom -> list of sample positions (into sampleIndices).
  // Using Map<number, number[]> rather than Int32Array buckets because the
  // sampled count per chromosome is modest (<<1024).
  const genesByChrom = new Map<number, number[]>();
  for (let i = 0; i < sampleIndices.length; i++) {
    const geneIdx = sampleIndices[i];
    const chrom = table.chroms[geneIdx];
    let list = genesByChrom.get(chrom);
    if (list === undefined) {
      list = [];
      genesByChrom.set(chrom, list);
    }
    list.push(geneIdx);
  }

  for (let k = 0; k < chromOrder.length; k++) {
    const chrom = chromOrder[k];
    if (chrom < 0) break; // padding sentinel

    const chromToken = tokens.chromosomeTokenMap[chrom];
    if (chromToken === undefined) {
      throw new Error(`no token mapping for chromosome ${chrom}`);
    }
    tokenIds[cursor] = chromToken;
    attentionMask[cursor] = 1;
    cursor++;

    const geneIdxs = genesByChrom.get(chrom);
    if (geneIdxs === undefined || geneIdxs.length === 0) {
      throw new Error(`chromosome ${chrom} listed in order but no genes sampled for it`);
    }
    // Sort ascending by genomic start. The sampler allows duplicates in
    // choice_idx (replace=True), and np.argsort is stable — do the same here.
    geneIdxs.sort((a, b) => {
      const sa = table.starts[a];
      const sb = table.starts[b];
      if (sa !== sb) return sa - sb;
      return a - b; // stable tiebreak on index
    });

    for (const geneIdx of geneIdxs) {
      tokenIds[cursor] = table.proteinIdsNew[geneIdx];
      attentionMask[cursor] = 1;
      cursor++;
    }

    tokenIds[cursor] = tokens.specials.chrom_token_right_idx;
    attentionMask[cursor] = 1;
    cursor++;
  }

  const pad = tokens.specials.pad_token_idx;
  for (let i = cursor; i < padLength; i++) {
    tokenIds[i] = pad;
    attentionMask[i] = 0;
  }

  return { tokenIds, attentionMask };
}
