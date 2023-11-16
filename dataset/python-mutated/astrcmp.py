def astrcmp_py(a, b):
    if False:
        i = 10
        return i + 15
    'Calculates the Levenshtein distance between a and b.'
    (n, m) = (len(a), len(b))
    if n > m:
        (a, b) = (b, a)
        (n, m) = (m, n)
    if n == 0 or m == 0.0:
        return 0.0
    current = range(n + 1)
    for i in range(1, m + 1):
        (previous, current) = (current, [i] + [0] * n)
        for j in range(1, n + 1):
            (add, delete) = (previous[j] + 1, current[j - 1] + 1)
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current[j] = min(add, delete, change)
    return 1.0 - current[n] / max(m, n)
try:
    from picard.util._astrcmp import astrcmp as astrcmp_c
    astrcmp = astrcmp_c
    astrcmp_implementation = 'C'
except ImportError:
    astrcmp = astrcmp_py
    astrcmp_implementation = 'Python'