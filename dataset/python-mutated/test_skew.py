import numpy as np
import pandas as pd
import pandas._testing as tm

def test_groupby_skew_equivalence():
    if False:
        return 10
    nrows = 1000
    ngroups = 3
    ncols = 2
    nan_frac = 0.05
    arr = np.random.default_rng(2).standard_normal((nrows, ncols))
    arr[np.random.default_rng(2).random(nrows) < nan_frac] = np.nan
    df = pd.DataFrame(arr)
    grps = np.random.default_rng(2).integers(0, ngroups, size=nrows)
    gb = df.groupby(grps)
    result = gb.skew()
    grpwise = [grp.skew().to_frame(i).T for (i, grp) in gb]
    expected = pd.concat(grpwise, axis=0)
    expected.index = expected.index.astype(result.index.dtype)
    tm.assert_frame_equal(result, expected)