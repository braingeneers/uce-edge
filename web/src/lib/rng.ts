/**
 * Tiny seedable RNG. Mulberry32 is a solid ~32-bit generator — good enough
 * for sampling; not cryptographic. Provides the same interface as Math.random
 * but with a seed for reproducibility.
 */
export class RNG {
  private state: number;

  constructor(seed: number) {
    // Mulberry32 requires non-zero seed; OR in a constant if 0 is passed.
    this.state = (seed | 0) || 0x9e3779b9;
  }

  /** Uniform float in [0, 1). */
  next(): number {
    let t = (this.state = (this.state + 0x6d2b79f5) | 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform int in [0, n). */
  nextInt(n: number): number {
    return Math.floor(this.next() * n);
  }

  /** Fisher-Yates shuffle in place. */
  shuffle<T>(arr: T[]): T[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = this.nextInt(i + 1);
      const tmp = arr[i];
      arr[i] = arr[j];
      arr[j] = tmp;
    }
    return arr;
  }
}
