def cronbach(itemscores):
    if False:
        while True:
            i = 10
    itemvars = itemscores.var(axis=1, ddof=1)
    tscores = itemscores.sum(axis=0)
    nitems = len(itemscores)
    return nitems / (nitems - 1) * (1 - itemvars.sum() / tscores.var(ddof=1))