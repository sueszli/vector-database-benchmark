def permutations(iterable, r=None):
    if False:
        while True:
            i = 10
    'permutations(range(3), 2) --> (0,1) (0,2) (1,0) (1,2) (2,0) (2,1)'
    pool = tuple(iterable)
    n = len(pool)
    if r is None:
        r = n
    indices = list(range(n))
    cycles = list(range(n - r + 1, n + 1))[::-1]
    yield tuple((pool[i] for i in indices[:r]))
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i + 1:] + indices[i:i + 1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                (indices[i], indices[-j]) = (indices[-j], indices[i])
                yield tuple((pool[i] for i in indices[:r]))
                break
        else:
            return

def n_queens(queen_count):
    if False:
        print('Hello World!')
    'N-Queens solver.\n    Args: queen_count: the number of queens to solve for, same as board size.\n    Yields: Solutions to the problem, each yielded value is a N-tuple.\n    '
    cols = range(queen_count)
    for vec in permutations(cols):
        if queen_count == len(set((vec[i] + i for i in cols))) == len(set((vec[i] - i for i in cols))):
            yield vec
bm_params = {(32, 10): (1, 5), (100, 25): (1, 6), (1000, 100): (1, 7), (5000, 100): (1, 8)}

def bm_setup(params):
    if False:
        for i in range(10):
            print('nop')
    res = None

    def run():
        if False:
            while True:
                i = 10
        nonlocal res
        for _ in range(params[0]):
            res = len(list(n_queens(params[1])))

    def result():
        if False:
            return 10
        return (params[0] * 10 ** (params[1] - 3), res)
    return (run, result)