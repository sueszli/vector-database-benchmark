"""
Tools for comparing ranked lists.
"""

def _rank_dists(ranks1, ranks2):
    if False:
        print('Hello World!')
    'Finds the difference between the values in ranks1 and ranks2 for keys\n    present in both dicts. If the arguments are not dicts, they are converted\n    from (key, rank) sequences.\n    '
    ranks1 = dict(ranks1)
    ranks2 = dict(ranks2)
    for k in ranks1:
        try:
            yield (k, ranks1[k] - ranks2[k])
        except KeyError:
            pass

def spearman_correlation(ranks1, ranks2):
    if False:
        while True:
            i = 10
    'Returns the Spearman correlation coefficient for two rankings, which\n    should be dicts or sequences of (key, rank). The coefficient ranges from\n    -1.0 (ranks are opposite) to 1.0 (ranks are identical), and is only\n    calculated for keys in both rankings (for meaningful results, remove keys\n    present in only one list before ranking).'
    n = 0
    res = 0
    for (k, d) in _rank_dists(ranks1, ranks2):
        res += d * d
        n += 1
    try:
        return 1 - 6 * res / (n * (n * n - 1))
    except ZeroDivisionError:
        return 0.0

def ranks_from_sequence(seq):
    if False:
        for i in range(10):
            print('nop')
    'Given a sequence, yields each element with an increasing rank, suitable\n    for use as an argument to ``spearman_correlation``.\n    '
    return ((k, i) for (i, k) in enumerate(seq))

def ranks_from_scores(scores, rank_gap=1e-15):
    if False:
        print('Hello World!')
    "Given a sequence of (key, score) tuples, yields each key with an\n    increasing rank, tying with previous key's rank if the difference between\n    their scores is less than rank_gap. Suitable for use as an argument to\n    ``spearman_correlation``.\n    "
    prev_score = None
    rank = 0
    for (i, (key, score)) in enumerate(scores):
        try:
            if abs(score - prev_score) > rank_gap:
                rank = i
        except TypeError:
            pass
        yield (key, rank)
        prev_score = score