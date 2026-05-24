#![forbid(unsafe_code)]
//! Entropy-efficient bounded random sampling.
//!
//! `EntropyPool` keeps an internal uniform state `(b, m)` and recycles the
//! parts of that state that a bounded draw does not consume. This avoids modulo
//! bias while allowing later draws to reuse residual entropy.
//!
//! # Example
//!
//! ```
//! use entropy_pool::EntropyPool;
//!
//! let mut pool = EntropyPool::new();
//!
//! let value = pool.gen_range(6);
//! assert!(value < 6);
//!
//! let permutation = pool.permutation(3, 10);
//! assert_eq!(permutation.len(), 3);
//!
//! let combination = pool.combination(5, 20);
//! assert_eq!(combination.len(), 5);
//! ```

use rand::{Rng, RngCore};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use wabi_tree::OSBTreeSet;

const BYTE_RADIX: u64 = 256;
const REJECTION_BOUND: u64 = 1u64 << 32;

/// Errors returned by the checked sampling methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntropyPoolError {
    /// `gen_range` was asked to sample from an empty range.
    EmptyRange,
    /// The requested sample is larger than the population.
    SampleTooLarge { sample: u32, population: u32 },
    /// The requested population cannot be represented or allocated on this
    /// platform.
    PopulationTooLarge { population: u32 },
    /// The byte counter overflowed.
    ByteCountOverflow,
}

impl fmt::Display for EntropyPoolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            EntropyPoolError::EmptyRange => write!(f, "range size must be greater than zero"),
            EntropyPoolError::SampleTooLarge { sample, population } => write!(
                f,
                "sample size {sample} is larger than population size {population}"
            ),
            EntropyPoolError::PopulationTooLarge { population } => {
                write!(f, "population size {population} is too large")
            }
            EntropyPoolError::ByteCountOverflow => write!(f, "random byte counter overflowed"),
        }
    }
}

impl Error for EntropyPoolError {}

/// A bounded sampler that reuses residual entropy between draws.
///
/// The default type parameter uses [`rand::rngs::ThreadRng`]. Use
/// [`EntropyPool::with_rng`] to provide a deterministic or custom RNG.
pub struct EntropyPool<R = rand::rngs::ThreadRng> {
    rng: R,
    b: u64,
    m: u64,
    count: u64,
}

impl EntropyPool<rand::rngs::ThreadRng> {
    /// Creates a new pool backed by [`rand::rng()`].
    pub fn new() -> Self {
        Self::with_rng(rand::rng())
    }
}

impl Default for EntropyPool<rand::rngs::ThreadRng> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: RngCore> EntropyPool<R> {
    /// Creates a new pool backed by `rng`.
    ///
    /// The pool reads one random byte immediately so that its initial state is
    /// uniform on `0..256`.
    pub fn with_rng(mut rng: R) -> Self {
        let b: u8 = rng.random();
        EntropyPool {
            rng,
            b: u64::from(b),
            m: BYTE_RADIX,
            count: 1,
        }
    }

    /// Returns the number of random bytes read from the backing RNG.
    pub fn random_bytes_read(&self) -> u64 {
        self.count
    }

    /// Returns the number of equiprobable states currently retained by the
    /// pool.
    pub fn retained_states(&self) -> u64 {
        self.m
    }

    /// Returns `log2(retained_states())`.
    pub fn retained_entropy_bits(&self) -> f64 {
        (self.m as f64).log2()
    }

    /// Returns a mutable reference to the backing RNG.
    pub fn rng_mut(&mut self) -> &mut R {
        &mut self.rng
    }

    /// Consumes the pool and returns the backing RNG.
    pub fn into_rng(self) -> R {
        self.rng
    }

    /// Returns a uniform integer in `0..n`.
    pub fn try_gen_range(&mut self, n: u32) -> Result<u32, EntropyPoolError> {
        if n == 0 {
            return Err(EntropyPoolError::EmptyRange);
        }

        let n = u64::from(n);
        loop {
            while (self.m % n) * REJECTION_BOUND >= self.m {
                self.append_random_byte()?;
            }

            let r = self.m % n;
            let q = self.m / n;
            if self.b < self.m - r {
                let b = self.b;
                self.m = q;
                self.b = b / n;
                return Ok((b % n) as u32);
            }

            self.b = self.m - self.b - 1;
            self.m = r;
        }
    }

    /// Returns a uniform integer in `0..n`.
    ///
    /// Panics if `n == 0`. Use [`Self::try_gen_range`] to handle invalid input
    /// without panicking.
    pub fn gen_range(&mut self, n: u32) -> u32 {
        self.try_gen_range(n)
            .expect("entropy-pool gen_range precondition failed")
    }

    /// Returns `m` distinct values from `0..n` in random order.
    pub fn try_permutation(&mut self, m: u32, n: u32) -> Result<Vec<u32>, EntropyPoolError> {
        if m > n {
            return Err(EntropyPoolError::SampleTooLarge {
                sample: m,
                population: n,
            });
        }
        if m == 0 {
            return Ok(Vec::new());
        }

        let take = usize::try_from(m)
            .map_err(|_| EntropyPoolError::PopulationTooLarge { population: n })?;
        let mut c = population_vec(n)?;
        for i in 0..m {
            let r = self.try_gen_range(n - i)? + i;
            c.swap(i as usize, r as usize);
        }
        c.truncate(take);
        Ok(c)
    }

    /// Returns `m` distinct values from `0..n` in random order.
    ///
    /// Panics if `m > n`. Use [`Self::try_permutation`] to handle invalid input
    /// without panicking.
    pub fn permutation(&mut self, m: u32, n: u32) -> Vec<u32> {
        self.try_permutation(m, n)
            .expect("entropy-pool permutation precondition failed")
    }

    /// Returns `m` distinct values from `0..n` as a sorted set.
    pub fn try_combination(&mut self, m: u32, n: u32) -> Result<BTreeSet<u32>, EntropyPoolError> {
        if m > n {
            return Err(EntropyPoolError::SampleTooLarge {
                sample: m,
                population: n,
            });
        }
        if m == 0 {
            return Ok(BTreeSet::new());
        }
        if m == n {
            return Ok(BTreeSet::from_iter(0..n));
        }

        let rev = m > n - m;
        let m = if rev { n - m } else { m };
        let mut s = OSBTreeSet::new();
        let mut c = population_vec(n)?;

        for i in (n - m..n).rev() {
            let r = self.try_gen_range(i + 1)?;
            let t = c[r as usize];
            c.swap(r as usize, i as usize);
            s.insert(t);

            let b = s.rank_of(&t).expect("inserted value should have a rank") as u32;
            self.recycle(b, n - i);
        }

        if !rev {
            Ok(BTreeSet::from_iter(s))
        } else {
            c.truncate((n - m) as usize);
            Ok(BTreeSet::from_iter(c))
        }
    }

    /// Returns `m` distinct values from `0..n` as a sorted set.
    ///
    /// Panics if `m > n`. Use [`Self::try_combination`] to handle invalid input
    /// without panicking.
    pub fn combination(&mut self, m: u32, n: u32) -> BTreeSet<u32> {
        self.try_combination(m, n)
            .expect("entropy-pool combination precondition failed")
    }

    fn append_random_byte(&mut self) -> Result<(), EntropyPoolError> {
        let Some(next_m) = self.m.checked_mul(BYTE_RADIX) else {
            return self.refill_full_width();
        };
        let next_count = self
            .count
            .checked_add(1)
            .ok_or(EntropyPoolError::ByteCountOverflow)?;
        let r: u8 = self.rng.random();

        self.b = self.b * BYTE_RADIX + u64::from(r);
        self.m = next_m;
        self.count = next_count;
        Ok(())
    }

    fn refill_full_width(&mut self) -> Result<(), EntropyPoolError> {
        loop {
            self.count = self
                .count
                .checked_add(8)
                .ok_or(EntropyPoolError::ByteCountOverflow)?;
            let b = self.rng.next_u64();
            if b != u64::MAX {
                self.b = b;
                self.m = u64::MAX;
                return Ok(());
            }
        }
    }

    fn recycle(&mut self, b: u32, m: u32) {
        debug_assert!(m > 0);
        debug_assert!(b < m);

        let b = u64::from(b);
        let m = u64::from(m);
        if let Some(next_m) = self.m.checked_mul(m) {
            self.b = self.b * m + b;
            self.m = next_m;
        } else {
            self.b = b;
            self.m = m;
        }
    }
}

fn population_vec(n: u32) -> Result<Vec<u32>, EntropyPoolError> {
    let capacity =
        usize::try_from(n).map_err(|_| EntropyPoolError::PopulationTooLarge { population: n })?;
    let mut c = Vec::new();
    c.try_reserve_exact(capacity)
        .map_err(|_| EntropyPoolError::PopulationTooLarge { population: n })?;
    c.extend(0..n);
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[derive(Clone)]
    struct ByteRng {
        next: u8,
    }

    impl ByteRng {
        fn new(seed: u8) -> Self {
            Self { next: seed }
        }

        fn next_byte(&mut self) -> u8 {
            let byte = self.next;
            self.next = self.next.wrapping_add(73);
            byte
        }
    }

    impl RngCore for ByteRng {
        fn next_u32(&mut self) -> u32 {
            let mut bytes = [0; 4];
            self.fill_bytes(&mut bytes);
            u32::from_le_bytes(bytes)
        }

        fn next_u64(&mut self) -> u64 {
            let mut bytes = [0; 8];
            self.fill_bytes(&mut bytes);
            u64::from_le_bytes(bytes)
        }

        fn fill_bytes(&mut self, dst: &mut [u8]) {
            for byte in dst {
                *byte = self.next_byte();
            }
        }
    }

    fn pool_with_state(b: u64, m: u64) -> EntropyPool {
        EntropyPool {
            rng: rand::rng(),
            b,
            m,
            count: 0,
        }
    }

    fn log2_choose(n: u64, k: u64) -> f64 {
        let k = k.min(n - k);
        (1..=k)
            .map(|i| ((n - k + i) as f64).log2() - (i as f64).log2())
            .sum()
    }

    #[test]
    fn custom_rng_makes_sampling_reproducible() {
        let rng = ByteRng::new(17);
        let mut a = EntropyPool::with_rng(rng.clone());
        let mut b = EntropyPool::with_rng(rng);

        assert_eq!(a.permutation(5, 20), b.permutation(5, 20));
        assert_eq!(a.combination(4, 12), b.combination(4, 12));
        assert_eq!(a.random_bytes_read(), b.random_bytes_read());
        assert_eq!(a.retained_states(), b.retained_states());
    }

    #[test]
    fn try_methods_report_invalid_arguments() {
        let mut ep = EntropyPool::new();

        assert_eq!(ep.try_gen_range(0), Err(EntropyPoolError::EmptyRange));
        assert_eq!(
            ep.try_permutation(3, 2),
            Err(EntropyPoolError::SampleTooLarge {
                sample: 3,
                population: 2
            })
        );
        assert_eq!(
            ep.try_combination(3, 2),
            Err(EntropyPoolError::SampleTooLarge {
                sample: 3,
                population: 2
            })
        );
    }

    #[test]
    fn recycle_overflow_discards_old_pool_without_biasing_state() {
        let mut ep = pool_with_state(0, u64::MAX);

        ep.recycle(1, 2);

        assert_eq!(ep.retained_states(), 2);
        assert_eq!(ep.b, 1);
        assert_eq!(ep.gen_range(2), 1);
        assert_eq!(ep.retained_states(), 1);
    }

    #[test]
    fn max_u32_range_uses_full_width_refill_without_u128() {
        let mut ep = EntropyPool::with_rng(ByteRng::new(11));

        let value = ep.gen_range(u32::MAX);

        assert!(value < u32::MAX);
        assert_eq!(ep.retained_states(), u64::from(u32::MAX) + 2);
    }

    #[test]
    fn permutation_2_of_3_is_exactly_uniform_when_pool_has_6_states() {
        let mut counts = BTreeMap::new();

        for seed in 0..6 {
            let mut ep = pool_with_state(seed, 6);
            let permutation = ep.permutation(2, 3);

            assert_eq!(
                ep.random_bytes_read(),
                0,
                "seed {seed} should not need extra bytes"
            );
            assert_eq!(
                ep.retained_states(),
                1,
                "seed {seed} should consume all entropy"
            );
            assert_eq!(ep.b, 0, "seed {seed} should leave the only residual state");
            *counts.entry(permutation).or_insert(0) += 1;
        }

        assert_eq!(counts.len(), 6);
        assert!(counts.values().all(|&count| count == 1), "{counts:#?}");
    }

    #[test]
    fn combination_2_of_4_is_uniform_and_recycles_order_entropy() {
        let mut observations = BTreeMap::new();

        for seed in 0..12 {
            let mut ep = pool_with_state(seed, 12);
            let combination = Vec::from_iter(ep.combination(2, 4));

            assert_eq!(
                ep.random_bytes_read(),
                0,
                "seed {seed} should not need extra bytes"
            );
            assert_eq!(
                ep.retained_states(),
                2,
                "seed {seed} should retain the discarded order bit"
            );
            assert!(ep.b < ep.m, "seed {seed} left invalid residual state");
            observations
                .entry(combination)
                .or_insert_with(Vec::new)
                .push(ep.b);
        }

        assert_eq!(observations.len(), 6);
        for (combination, mut residuals) in observations {
            residuals.sort_unstable();
            assert_eq!(
                residuals,
                vec![0, 1],
                "combination {combination:?} should appear once per recycled order state"
            );
        }
    }

    #[test]
    fn combination_handles_edges_and_complements() {
        let mut ep = EntropyPool::new();

        assert_eq!(ep.combination(0, 10), BTreeSet::new());
        assert_eq!(
            ep.combination(10, 10),
            BTreeSet::from_iter(0..10),
            "selecting the full population should return every item"
        );

        for (m, n) in [(1, 10), (3, 10), (7, 10), (9, 10)] {
            let combination = ep.combination(m, n);
            assert_eq!(combination.len(), m as usize);
            assert!(
                combination.iter().all(|&value| value < n),
                "combination({m}, {n}) returned an out-of-range value: {combination:?}"
            );
        }
    }

    #[test]
    fn large_combination_uses_entropy_near_the_binomial_limit() {
        let selected: u32 = 20_000;
        let population: u32 = 30_000;
        let mut ep = EntropyPool::new();

        let combination = ep.combination(selected, population);
        let minimum_bits = log2_choose(u64::from(population), u64::from(selected));
        let consumed_bits = ep.random_bytes_read() as f64 * 8.0;
        let retained_bits = ep.retained_entropy_bits();
        let output_efficiency = minimum_bits / consumed_bits;
        let retained_efficiency = (minimum_bits + retained_bits) / consumed_bits;

        assert_eq!(combination.len(), selected as usize);
        assert!(combination.iter().all(|&value| value < population));
        println!(
            "combination({selected}, {population}): lower_bound={minimum_bits:.2} bits, \
             retained={retained_bits:.2} bits, consumed={consumed_bits:.0} bits \
             ({} bytes), output_efficiency={:.4}%, retained_efficiency={:.4}%",
            ep.random_bytes_read(),
            output_efficiency * 100.0,
            retained_efficiency * 100.0
        );

        assert!(
            output_efficiency > 0.995,
            "entropy efficiency was {:.4}%, lower_bound={minimum_bits:.2}, consumed={consumed_bits:.0}",
            output_efficiency * 100.0
        );
    }
}
