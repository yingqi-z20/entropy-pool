use rand::Rng;
use std::collections::BTreeSet;
use wabi_tree::OSBTreeSet;

pub struct EntropyPool {
    rng: rand::rngs::ThreadRng,
    b: u64,
    m: u64,
    count: u64,
}

impl EntropyPool {
    pub fn new() -> EntropyPool {
        let mut rng = rand::rng();
        let b: u8 = rng.random();
        let m: u64 = 256;
        let count: u64 = 1;
        EntropyPool {
            rng,
            b: b as u64,
            m,
            count,
        }
    }

    pub fn gen_range(&mut self, n: u64) -> u64 {
        loop {
            while 4294967296u64 * (self.m % n) >= self.m {
                let r: u8 = self.rng.random();
                self.count += 1;
                self.m *= 256;
                self.b = self.b * 256 + r as u64;
            }
            let r = self.m % n;
            let q = self.m / n;
            if r < self.m - self.b {
                let b = self.b;
                self.m = q;
                self.b = b / n;
                return b % n;
            } else {
                self.b = self.m - self.b - 1;
                self.m = r;
            }
        }
    }

    fn recycle(&mut self, b: u64, m: u64) {
        self.b = self.b * m + b;
        self.m = self.m * m;
    }

    pub fn permutation(&mut self, m: u64, n: u64) -> Vec<u64> {
        let mut c = Vec::from_iter(0..n);
        for i in 0..m {
            let r = self.gen_range(n - i) + i;
            let tmp = c[i as usize];
            c[i as usize] = c[r as usize];
            c[r as usize] = tmp;
        }
        c.truncate(m as usize);
        c
    }

    pub fn combination(&mut self, m: u64, n: u64) -> BTreeSet<u64> {
        let rev = 2 * m > n;
        let m = if rev { n - m } else { m };
        let mut s = OSBTreeSet::with_capacity(m as usize);
        let mut c = Vec::from_iter(0..n);
        for i in (n - m..n).rev() {
            let r = self.gen_range(i + 1);
            let t = c[r as usize];
            c[r as usize] = c[i as usize];
            s.insert(t);
            let b = s.rank_of(&t).unwrap() as u64;
            self.recycle(b, n - i);
        }
        if !rev {
            BTreeSet::from_iter(s.into_iter())
        } else {
            c.truncate((n - m) as usize);
            BTreeSet::from_iter(c.into_iter())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    #[test]
    fn it_works() {
        let mut ep = EntropyPool::new();
        assert_eq!(ep.combination(200000, 300000).len(), 200000);
        println!("{:#?}", ep.count);
        ep = EntropyPool::new();
        let mut m = BTreeMap::new();
        for _ in 0..1000000 {
            m.entry(ep.combination(2, 6))
                .and_modify(|e| *e += 1)
                .or_insert(1);
        }
        println!("{:#?}", m.into_values());
    }
}
