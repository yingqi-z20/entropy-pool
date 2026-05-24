# entropy-pool

`entropy-pool` is a small Rust crate for drawing bounded random values,
partial permutations, and combinations while keeping unused randomness in an
internal entropy pool.

The main idea is simple: every random byte that is read expands an internal
uniform state space. When a draw cannot use that space exactly, the leftover
states are not thrown away. They are mapped back into the pool and reused by
later draws.

## Example

```rust
use entropy_pool::EntropyPool;

let mut pool = EntropyPool::new();

// Uniform value in 0..6.
let die_roll = pool.gen_range(6) + 1;

// Three distinct values from 0..10, in random order.
let sample = pool.permutation(3, 10);

// Five distinct values from 0..20, returned as a sorted BTreeSet.
let chosen = pool.combination(5, 20);
```

## API

### `EntropyPool::new()`

Creates a new pool backed by `rand::rng()`. The pool starts by reading one
random byte.

### `EntropyPool::with_rng(rng)`

Creates a new pool backed by a caller-provided RNG implementing
`rand::RngCore`. This is useful for deterministic tests, seeded generators, or
specialized randomness sources.

### `gen_range(n: u32) -> u32`

Returns a uniform integer in `0..n`.

Panics if `n == 0`.

Use `try_gen_range(n)` to get `Result<u32, EntropyPoolError>` instead.

### `permutation(m: u32, n: u32) -> Vec<u32>`

Returns `m` distinct values from `0..n` in random order.

This is a partial Fisher-Yates shuffle driven by `gen_range`.

Panics if `m > n`.

Use `try_permutation(m, n)` to get `Result<Vec<u32>, EntropyPoolError>`
instead.

### `combination(m: u32, n: u32) -> BTreeSet<u32>`

Returns `m` distinct values from `0..n` as a sorted set.

For large selections, the implementation samples the smaller side and returns
the complement, so `combination(90, 100)` only has to sample 10 excluded values.

Panics if `m > n`.

Use `try_combination(m, n)` to get `Result<BTreeSet<u32>, EntropyPoolError>`
instead.

### Statistics

`random_bytes_read()` returns how many bytes have been read from the backing
RNG. `retained_states()` returns the current pool radix `m`, and
`retained_entropy_bits()` returns `log2(m)`.

## How The Entropy Pool Works

The pool stores a pair `(b, m)` with this invariant:

```text
b is uniformly distributed in 0..m
```

To draw a value in `0..n`, the algorithm writes:

```text
m = q * n + r
```

If `b` falls in the first `q * n` states, the draw is exact:

```text
output = b % n
new_pool = (b / n, q)
```

If `b` falls in the remaining `r` states, those rejected states are converted
back into a uniform state in `0..r` and the draw continues. Before each trial,
the pool is expanded until the rejection probability is below `2^-32`. While
the internal `u64` state can hold the expanded radix, this appends one byte:

```text
new_b = old_b * 256 + random_byte
new_m = old_m * 256
```

If another byte would overflow the `u64` radix and the rejection bound is still
not met, the implementation discards the old retained state and refills the
pool with a full-width uniform state on `0..u64::MAX`.

This is a byte-oriented entropy recycling method: modulo bias is avoided, but
the rejection branch keeps the unused entropy instead of discarding it.

## Mathematical Proof

Write `[k]` for the set `{0, 1, ..., k - 1}`. The panicking methods require the
documented preconditions; the `try_*` methods report invalid arguments with
`EntropyPoolError`.

### Entropy-pool invariant

At every public draw boundary, the pool state `(B, M)` satisfies:

```text
B is uniform on [M].
```

Future bytes produced by `rand::rng()` are independent of `B`.

When the implementation appends one random byte `U`, where `U` is uniform on
`[256]`, it replaces:

```text
B' = 256B + U
M' = 256M
```

The map

```text
(B, U) -> 256B + U
```

is a bijection from `[M] x [256]` to `[256M]`. Therefore `B'` is uniform on
`[M']`, so appending bytes preserves the invariant.

If appending another byte would overflow the internal `u64` radix, the
implementation may instead draw a `u64` value `X` until `X != u64::MAX`, then
store:

```text
B' = X
M' = u64::MAX
```

Conditioned on `X != u64::MAX`, `X` is uniform on `[u64::MAX]`. This full-width
refill discards older retained entropy, but it preserves the invariant and
keeps the `u32` range sampler on fast `u64` arithmetic.

### Correctness of `gen_range`

Let the current pool state be `(B, M)`, with `B` uniform on `[M]`. For a
requested range size `n > 0`, write:

```text
M = qn + r, 0 <= r < n.
```

The algorithm accepts when:

```text
B < qn.
```

On the accepted set `[qn]`, define:

```text
Y = B mod n
C = floor(B / n)
```

The map

```text
B -> (Y, C)
```

is a bijection from `[qn]` to `[n] x [q]`, with inverse:

```text
(Y, C) -> Cn + Y.
```

Conditioned on acceptance, `B` is uniform on `[qn]`; by the bijection,
`(Y, C)` is uniform on `[n] x [q]`. Hence:

- `Y` is uniform on `[n]`;
- `C` is uniform on `[q]`;
- `Y` and `C` are independent.

The implementation returns `Y` and stores `(C, q)` as the new pool state, so the
returned value is unbiased and the pool invariant is preserved.

If the algorithm rejects, then `B` lies in:

```text
{qn, qn + 1, ..., M - 1},
```

which has size `r`. The implementation stores:

```text
D = M - B - 1
```

and `M = r`. The map `B -> D` is a bijection from the rejected set to `[r]`.
Conditioned on rejection, `D` is therefore uniform on `[r]`, so the pool
invariant is again preserved and the loop can retry. No modulo-biased value is
ever returned.

The byte-extension loop is separate from the acceptance bijection above: it
ensures there is an accepted region and bounds the rejection probability. Once
the loop condition is false:

```text
2^32 * (M mod n) < M,
```

so the next trial rejects with probability:

```text
r / M < 2^-32.
```

Thus, within the finite-size limits of this implementation, repeated rejection
has probability zero in the ideal random-byte model.

### Correctness of `permutation`

`permutation(m, n)` is a partial Fisher-Yates shuffle. At step `i`, the suffix
before `i` is already fixed, and the array positions `i..n` contain exactly the
remaining `n - i` items. The call:

```text
gen_range(n - i) + i
```

chooses one of those remaining positions uniformly, and the swap moves that
chosen item into position `i`.

By the `gen_range` proof, each choice is uniform and leaves a pool state
independent of the chosen index. Therefore the probability of any ordered
`m`-tuple of distinct values is:

```text
1 / n * 1 / (n - 1) * ... * 1 / (n - m + 1)
= 1 / P(n, m),
```

where:

```text
P(n, m) = n! / (n - m)!.
```

So `permutation(m, n)` is uniform over all ordered length-`m` samples without
replacement.

### Correctness of `recycle`

Suppose the current pool state is `(B, M)`, with `B` uniform on `[M]`. Suppose
`D` is independent of `B` and uniform on `[d]`. `recycle(D, d)` stores:

```text
B' = Bd + D
M' = Md
```

The map:

```text
(B, D) -> Bd + D
```

is a bijection from `[M] x [d]` to `[Md]`. Therefore `B'` is uniform on `[M']`.
So, when `Md` fits in the internal `u64` state, recycling an independent
base-`d` digit preserves the pool invariant while adding exactly `log2(d)` bits
of state space.

If `Md` would overflow `u64`, the implementation discards the older retained
state and stores only:

```text
B' = D
M' = d
```

Since `D` is uniform on `[d]`, the pool invariant is still preserved. This
fallback can lose retained entropy, but it cannot bias future outputs.

### Correctness of `combination`

A combination has less entropy than an ordered sample. For example, choosing
two items from four has:

```text
C(4, 2) = 6 combinations
P(4, 2) = 12 ordered samples
```

The implementation samples through a shuffle-like process, but after each item
is inserted into an order-statistics tree it recycles the item's rank among the
already-selected items. That rank is exactly the ordering information that a
combination does not need.

So the final set remains uniformly distributed, while the discarded ordering
entropy is returned to the pool for later draws.

For the proof, let `k` be the number of items actually sampled by the
implementation:

```text
k = min(m, n - m).
```

First ignore the complement optimization and consider sampling `k` selected
items. The loop chooses an ordered sequence:

```text
T = (T1, T2, ..., Tk)
```

without replacement from `[n]`. By the same argument as `permutation`, `T` is
uniform over the `P(n, k)` ordered distinct sequences. Repeated application of
the `gen_range` proof also gives that the pool state remaining after these
draws is independent of `T`.

Let:

```text
S_j = {T1, T2, ..., Tj}.
```

After inserting `Tj`, the implementation computes:

```text
R_j = rank of Tj inside S_j,
```

so `R_j` is an integer in `[j]`.

Now consider the map:

```text
T -> (S_k, R_1, R_2, ..., R_k).
```

This map is a bijection.

To see injectivity, reconstruct `T` backwards. Given the final set `S_k` and
the ranks `R_1..R_k`, `T_k` is the `R_k`-th smallest element of `S_k`. Remove
it to recover `S_{k-1}`. Then `T_{k-1}` is the `R_{k-1}`-th smallest element of
`S_{k-1}`. Continue until `T_1` is recovered. Hence no two ordered sequences
produce the same `(S_k, R_1, ..., R_k)`.

The codomain has size:

```text
C(n, k) * 1 * 2 * ... * k
= C(n, k) * k!
= P(n, k),
```

which equals the number of possible ordered sequences. Therefore the injective
map is bijective.

Since `T` is uniform and the bijection factors it into:

```text
S_k in the C(n, k) possible sets
(R_1, ..., R_k) in [1] x [2] x ... x [k],
```

the set `S_k` is uniform over all `k`-subsets, and the rank tuple is uniform
and independent of `S_k`.

After each insertion, the implementation calls:

```text
recycle(R_j, j)
```

The rank tuple is exactly the ordering information that distinguishes an
ordered sample from its unordered set. Because the tuple is independent of the
final set, each recycled rank digit preserves the pool invariant conditioned on
the returned combination. If all recycle multiplications fit, after all `k`
steps the pool has recovered:

```text
log2(1 * 2 * ... * k) = log2(k!)
```

bits of state space, up to integer radix representation. If a recycle overflow
fallback occurs, correctness is unchanged, but some previously retained entropy
is deliberately discarded.

Therefore `combination(k, n)` returns a uniform `k`-subset and recycles the
discarded order entropy.

When the requested `m` is larger than `n / 2`, the implementation samples the
excluded set of size `n - m` and returns its complement. The complement map is a
bijection between `(n - m)`-subsets and `m`-subsets, so uniformity is preserved.
The recycled rank tuple is independent of the excluded set, and therefore also
independent of its complement.

### Entropy lower bound

Any exact sampler for `m` elements from `n` must produce one of:

```text
C(n, m)
```

outputs uniformly. Thus the output itself contains:

```text
log2 C(n, m)
```

bits of entropy. The tests compare this information-theoretic lower bound with
the number of random bytes actually read. For a pool state `(B, M)`, the
remaining reusable entropy is:

```text
log2 M.
```

So the useful entropy after a combination draw is measured as:

```text
log2 C(n, m) + log2 M.
```

divided by the consumed random bits. Values close to `1` mean the consumed
randomness has been converted almost entirely into either the returned
combination or reusable pool state.

## Correctness And Entropy Tests

Run the test suite with:

```sh
cargo test -- --nocapture
```

The tests include:

- deterministic custom-RNG sampling;
- checked error handling for invalid arguments;
- `u64` recycle-overflow fallback;
- exhaustive uniformity for `permutation(2, 3)`;
- exhaustive uniformity for `combination(2, 4)`;
- verification that `combination(2, 4)` recycles the discarded order bit;
- edge cases for empty, full, direct, and complement combinations;
- a large entropy-utilization check against the binomial lower bound.

Example output from the large test:

```text
combination(20000, 30000): lower_bound=27541.20 bits, retained=50.80 bits,
consumed=27592 bits (3449 bytes), output_efficiency=99.8159%,
retained_efficiency=100.0000%
```

The exact byte count can vary with random rejection paths, but the test asserts
that output entropy stays above 99.5% of the information-theoretic lower bound.
When retained pool entropy is included, the accounting should be very close to
100%.

## Notes

- The default pool uses `ThreadRng`, but `EntropyPool::with_rng` accepts any
  `rand::RngCore`.
- Public sampling APIs use `u32` ranges and the internal pool state is `u64`.
  This keeps the hot path on native-width arithmetic. If expansion or recycling
  cannot keep older retained entropy inside `u64`, the implementation keeps
  correctness by discarding that older retained entropy.
- `permutation` and `combination` currently build a `Vec<u32>` for the whole
  population, so memory use is `O(n)`.
- The non-`try_*` methods are convenience wrappers that panic on invalid
  arguments. Use the checked methods for library-facing error handling.

## License

AGPL-3.0-only
