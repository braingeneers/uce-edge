/**
 * Weighted random sampling with replacement — browser equivalent of
 * `np.random.Generator.choice(p=weights, replace=True)`.
 *
 * Strategy: cumulative sum + binary search. O(G) to build, O(log G) per draw.
 * For G=18k and 1024 draws, this is ~14k comparisons — trivial.
 *
 * The numerical result will NOT match numpy bit-for-bit because Mulberry32 !=
 * numpy's PCG64; phase 5 validation uses cosine / distributional checks.
 */
import { RNG } from "./rng.js";

export function buildCumulativeWeights(weights: Float32Array): Float64Array {
  // Use Float64 for the prefix sum to avoid catastrophic precision loss at
  // 18k terms. Final normalization tolerates weights that don't sum to exactly
  // 1.0 (they won't, coming from Float32 log1p/divide).
  const cum = new Float64Array(weights.length);
  let acc = 0;
  for (let i = 0; i < weights.length; i++) {
    acc += weights[i];
    cum[i] = acc;
  }
  return cum;
}

/** Binary search: return smallest i such that cum[i] >= target. */
function searchCumulative(cum: Float64Array, target: number): number {
  let lo = 0;
  let hi = cum.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (cum[mid] < target) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

/**
 * Sample `sampleSize` indices from [0, G) according to `weights`, with
 * replacement. Returns Int32Array of length sampleSize.
 */
export function weightedSample(
  weights: Float32Array,
  sampleSize: number,
  rng: RNG
): Int32Array {
  const cum = buildCumulativeWeights(weights);
  const total = cum[cum.length - 1];
  if (!(total > 0)) {
    throw new Error(`weights sum to non-positive total: ${total}`);
  }
  const out = new Int32Array(sampleSize);
  for (let i = 0; i < sampleSize; i++) {
    const u = rng.next() * total;
    out[i] = searchCumulative(cum, u);
  }
  return out;
}
