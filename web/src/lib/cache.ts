/**
 * Durable fetch via the Cache API.
 *
 * First call downloads and stores the response. Subsequent calls (same URL,
 * even across reloads) hit the cache directly. Good for the ~400 MB protein
 * embedding table.
 */

const CACHE_NAME = "uce-edge-v1";

export interface CachedFetchResult {
  buffer: ArrayBuffer;
  fromCache: boolean;
  ms: number;
}

export async function cachedFetchBytes(url: string): Promise<CachedFetchResult> {
  const t0 = performance.now();

  // Cache API is unavailable in some contexts (insecure origin, certain
  // Chromium launch modes). Fall through to plain fetch when that happens.
  let cache: Cache | null = null;
  try {
    cache = await caches.open(CACHE_NAME);
  } catch {
    cache = null;
  }

  if (cache) {
    const hit = await cache.match(url);
    if (hit) {
      const buffer = await hit.arrayBuffer();
      return { buffer, fromCache: true, ms: performance.now() - t0 };
    }
  }

  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status}`);

  if (cache) {
    // Response body can only be consumed once — clone before caching.
    // `put` can fail on large bodies in some environments (quota, internal
    // limits). Treat it as best-effort.
    try {
      await cache.put(url, resp.clone());
    } catch (e) {
      console.warn(`cache.put failed for ${url}: ${(e as Error).message}`);
    }
  }

  const buffer = await resp.arrayBuffer();
  return { buffer, fromCache: false, ms: performance.now() - t0 };
}
