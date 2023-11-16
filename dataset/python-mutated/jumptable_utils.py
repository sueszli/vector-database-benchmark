import math
from dataclasses import dataclass
from vyper.utils import method_id_int

@dataclass
class Signature:
    method_id: int
    payable: bool

@dataclass
class Bucket:
    bucket_id: int
    magic: int
    method_ids: list[int]

    @property
    def image(self):
        if False:
            return 10
        return _image_of([s for s in self.method_ids], self.magic)

    @property
    def method_ids_image_order(self):
        if False:
            while True:
                i = 10
        return [x[1] for x in sorted(zip(self.image, self.method_ids))]

    @property
    def bucket_size(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.method_ids)
BITS_MAGIC = 24

def _image_of(xs, magic):
    if False:
        i = 10
        return i + 15
    bits_shift = BITS_MAGIC
    return [(x * magic >> bits_shift) % len(xs) for x in xs]

class _FindMagicFailure(Exception):
    pass

class _HasEmptyBuckets(Exception):
    pass

def find_magic_for(xs):
    if False:
        for i in range(10):
            print('nop')
    for m in range(2 ** 16):
        test = _image_of(xs, m)
        if len(test) == len(set(test)):
            return m
    raise _FindMagicFailure(f'Could not find hash for {xs}')

def _mk_buckets(method_ids, n_buckets):
    if False:
        i = 10
        return i + 15
    buckets = {}
    for x in method_ids:
        t = x % n_buckets
        buckets.setdefault(t, [])
        buckets[t].append(x)
    return buckets

def _dense_jumptable_info(method_ids, n_buckets):
    if False:
        return 10
    buckets = _mk_buckets(method_ids, n_buckets)
    if len(buckets) != n_buckets:
        raise _HasEmptyBuckets()
    ret = {}
    for (bucket_id, method_ids) in buckets.items():
        magic = find_magic_for(method_ids)
        ret[bucket_id] = Bucket(bucket_id, magic, method_ids)
    return ret
START_BUCKET_SIZE = 5

def generate_dense_jumptable_info(signatures):
    if False:
        while True:
            i = 10
    method_ids = [method_id_int(sig) for sig in signatures]
    n = len(signatures)
    n_buckets = n // START_BUCKET_SIZE + 1
    ret = None
    tried_exhaustive = False
    while n_buckets > 0:
        try:
            solution = _dense_jumptable_info(method_ids, n_buckets)
            assert len(solution) == n_buckets
            ret = (n_buckets, solution)
        except _HasEmptyBuckets:
            pass
        except _FindMagicFailure:
            if ret is not None:
                break
            if not tried_exhaustive:
                n_buckets = n
                tried_exhaustive = True
                continue
            else:
                raise RuntimeError(f'Could not generate jumptable! {signatures}')
        n_buckets -= 1
    return ret

def generate_sparse_jumptable_buckets(signatures):
    if False:
        while True:
            i = 10
    method_ids = [method_id_int(sig) for sig in signatures]
    n = len(signatures)
    lo = max(1, math.floor(n * 0.85))
    hi = max(1, math.ceil(n * 1.15))
    stats = {}
    for i in range(lo, hi + 1):
        buckets = _mk_buckets(method_ids, i)
        stats[i] = buckets
    min_max_bucket_size = hi + 1
    for (i, buckets) in stats.items():
        max_bucket_size = max((len(bucket) for bucket in buckets.values()))
        if max_bucket_size < min_max_bucket_size:
            min_max_bucket_size = max_bucket_size
            ret = (i, buckets)
    assert ret is not None
    return ret

def _bench_dense(N=1000, n_methods=100):
    if False:
        return 10
    import random
    stats = []
    for i in range(N):
        seed = random.randint(0, 2 ** 64 - 1)
        sigs = [f'foo{i + seed}()' for i in range(n_methods)]
        xs = generate_dense_jumptable_info(sigs)
        print(f'found. n buckets {len(xs)}')
        stats.append(xs)

    def mean(xs):
        if False:
            while True:
                i = 10
        return sum(xs) / len(xs)
    avg_n_buckets = mean([len(jt) for jt in stats])
    print(f'average N buckets: {avg_n_buckets}')

def _bench_sparse(N=10000, n_methods=80):
    if False:
        i = 10
        return i + 15
    import random
    stats = []
    for _ in range(N):
        seed = random.randint(0, 2 ** 64 - 1)
        sigs = [f'foo{i + seed}()' for i in range(n_methods)]
        (_, buckets) = generate_sparse_jumptable_buckets(sigs)
        bucket_sizes = [len(bucket) for bucket in buckets.values()]
        worst_bucket_size = max(bucket_sizes)
        mean_bucket_size = sum(bucket_sizes) / len(bucket_sizes)
        stats.append((worst_bucket_size, mean_bucket_size))
    print('worst worst bucket size:', max((x[0] for x in stats)))
    print('avg worst bucket size:', sum((x[0] for x in stats)) / len(stats))
    print('worst mean bucket size:', max((x[1] for x in stats)))
    print('avg mean bucket size:', sum((x[1] for x in stats)) / len(stats))