from boltons.statsutils import Stats

def test_stats_basic():
    if False:
        print('Hello World!')
    da = Stats(range(20))
    assert da.mean == 9.5
    assert round(da.std_dev, 2) == 5.77
    assert da.variance == 33.25
    assert da.skewness == 0
    assert round(da.kurtosis, 1) == 1.9
    assert da.median == 9.5

def _test_pearson():
    if False:
        for i in range(10):
            print('nop')
    import random
    from statsutils import pearson_type

    def get_pt(dist):
        if False:
            return 10
        vals = [dist() for x in range(10000)]
        pt = pearson_type(vals)
        return pt
    for x in range(3):
        pt = get_pt(dist=lambda : random.uniform(0.0, 10.0))
        print('pearson type:', pt)