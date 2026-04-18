/**
 * Load Phase 1 reference fixtures produced by
 * `scripts/brain_reference_pipeline.py`. All bin files are flat dense arrays
 * in row-major order; shapes/dtypes come from manifest.json.
 */

export interface FixtureManifest {
  n_cells: number;
  n_aligned_genes: number;
  pad_length: number;
  sample_size: number;
  embedding_dim: number;
  d_model: number;
  files: Record<string, { shape: number[]; dtype: string }>;
}

export async function fetchBytes(url: string): Promise<ArrayBuffer> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);
  return resp.arrayBuffer();
}

export async function fetchJSON<T>(url: string): Promise<T> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);
  return resp.json() as Promise<T>;
}

export async function loadFloat32(url: string, expectedLen?: number): Promise<Float32Array> {
  const buf = await fetchBytes(url);
  const arr = new Float32Array(buf);
  if (expectedLen !== undefined && arr.length !== expectedLen) {
    throw new Error(`${url}: expected ${expectedLen} floats, got ${arr.length}`);
  }
  return arr;
}

export async function loadInt32(url: string, expectedLen?: number): Promise<Int32Array> {
  const buf = await fetchBytes(url);
  const arr = new Int32Array(buf);
  if (expectedLen !== undefined && arr.length !== expectedLen) {
    throw new Error(`${url}: expected ${expectedLen} ints, got ${arr.length}`);
  }
  return arr;
}

export function productOf(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}
