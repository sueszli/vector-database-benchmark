from functools import wraps

def recurrence_memo(initial):
    if False:
        for i in range(10):
            print('nop')
    '\n    Memo decorator for sequences defined by recurrence\n\n    See usage examples e.g. in the specfun/combinatorial module\n    '
    cache = initial

    def decorator(f):
        if False:
            i = 10
            return i + 15

        @wraps(f)
        def g(n):
            if False:
                while True:
                    i = 10
            L = len(cache)
            if n <= L - 1:
                return cache[n]
            for i in range(L, n + 1):
                cache.append(f(i, cache))
            return cache[-1]
        return g
    return decorator

def assoc_recurrence_memo(base_seq):
    if False:
        while True:
            i = 10
    '\n    Memo decorator for associated sequences defined by recurrence starting from base\n\n    base_seq(n) -- callable to get base sequence elements\n\n    XXX works only for Pn0 = base_seq(0) cases\n    XXX works only for m <= n cases\n    '
    cache = []

    def decorator(f):
        if False:
            i = 10
            return i + 15

        @wraps(f)
        def g(n, m):
            if False:
                for i in range(10):
                    print('nop')
            L = len(cache)
            if n < L:
                return cache[n][m]
            for i in range(L, n + 1):
                F_i0 = base_seq(i)
                F_i_cache = [F_i0]
                cache.append(F_i_cache)
                for j in range(1, i + 1):
                    F_ij = f(i, j, cache)
                    F_i_cache.append(F_ij)
            return cache[n][m]
        return g
    return decorator