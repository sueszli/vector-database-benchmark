try:
    import random
except ImportError:
    print('SKIP')
    raise SystemExit
try:
    random.seed()
except ValueError:
    print('SKIP')
    raise SystemExit

def rng_seq():
    if False:
        i = 10
        return i + 15
    return [random.getrandbits(16) for _ in range(10)]
random.seed()
seq = rng_seq()
random.seed()
print(seq == rng_seq())
random.seed(None)
print(seq == rng_seq())