/**
 * Stage 2: per-cell log1p + sum-to-1 normalize. Matches the Python reference:
 *
 *   log_expr = torch.log1p(counts)                 # elementwise
 *   sum = clamp(log_expr.sum(dim=1, keepdim=True), min=1e-8)
 *   weights = log_expr / sum
 *
 * Pure elementwise + reduction — deterministic, so JS and Python should match
 * to Float32 roundoff (~1e-7). We compute the sum in Float64 to match PyTorch's
 * accumulator precision for a reduction over ~18k terms.
 */
export function normalizeCounts(counts: Float32Array): Float32Array {
  const G = counts.length;
  const weights = new Float32Array(G);
  let sum = 0;
  for (let i = 0; i < G; i++) {
    const v = Math.log1p(counts[i]);
    weights[i] = v;
    sum += v;
  }
  const denom = Math.max(sum, 1e-8);
  for (let i = 0; i < G; i++) weights[i] = weights[i] / denom;
  return weights;
}
