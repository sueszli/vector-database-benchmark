"""Determine new partition divisions using approximate percentiles.

We use a custom algorithm to calculate approximate, evenly-distributed
percentiles of arbitrarily-ordered data for any dtype in a distributed
fashion with one pass over the data.  This is used to determine new
partition divisions when changing the index of a dask.dataframe.  We claim
no statistical guarantees, but we use a variety of heuristics to try to
provide reliable, robust results that are "good enough" and can scale to
large number of partitions.

Our approach is similar to standard approaches such as t- and q-digest,
GK, and sampling-based algorithms, which consist of three parts:

1. **Summarize:** create summaries of subsets of data
2. **Merge:** combine summaries to make a new summary
3. **Compress:** periodically compress a summary into a smaller summary

We summarize the data in each partition by calculating several percentiles.
The value at each percentile is given a weight proportional to the length
of the partition and the differences between the current percentile and
the adjacent percentiles.  Merging summaries is simply a ``merge_sorted``
of the values and their weights, which we do with a reduction tree.

Percentiles is a good choice for our case, because we are given a numpy
array of the partition's data, and percentiles is a relatively cheap
operation.  Moreover, percentiles are, by definition, much less
susceptible to the underlying distribution of the data, so the weights
given to each value--even across partitions--should be comparable.

Let us describe this to a child of five.  We are given many small cubes
(of equal size) with numbers on them.  Split these into many piles.  This
is like the original data.  Let's sort and stack the cubes from one of the
piles.  Next, we are given a bunch of unlabeled blocks of different sizes,
and most are much larger than the the original cubes.  Stack these blocks
until they're the same height as our first stack.  Let's write a number on
each block of the new stack.  To do this, choose the number of the cube in
the first stack that is located in the middle of an unlabeled block.  We
are finished with this stack once all blocks have a number written on them.
Repeat this for all the piles of cubes.  Finished already?  Great!  Now
take all the stacks of the larger blocks you wrote on and throw them into
a single pile.  We'll be sorting these blocks next, which may be easier if
you carefully move the blocks over and organize... ah, nevermind--too late.
Okay, sort and stack all the blocks from that amazing, disorganized pile
you just made.  This will be very tall, so we had better stack it sideways
on the floor like so.  This will also make it easier for us to split the
stack into groups of approximately equal size, which is our final task...

This, in a nutshell, is the algorithm we deploy.  The main difference
is that we don't always assign a block the number at its median (ours
fluctuates around the median).  The numbers at the edges of the final
groups is what we use as divisions for repartitioning.  We also need
the overall min and max, so we take the 0th and 100th percentile of
each partition, and another sample near each edge so we don't give
disproportionate weights to extreme values.

Choosing appropriate percentiles to take in each partition is where things
get interesting.  The data is arbitrarily ordered, which means it may be
sorted, random, or follow some pathological distribution--who knows.  We
hope all partitions are of similar length, but we ought to expect some
variation in lengths.  The number of partitions may also be changing
significantly, which could affect the optimal choice of percentiles.  For
improved robustness, we use both evenly-distributed and random percentiles.
If the number of partitions isn't changing, then the total number of
percentiles across all partitions scales as ``npartitions**1.5``.  Although
we only have a simple compression operation (step 3 above) that combines
weights of equal values, a more sophisticated one could be added if needed,
such as for extremely large ``npartitions`` or if we find we need to
increase the sample size for each partition.

"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_dtype, is_integer_dtype
from tlz import merge, merge_sorted, take
from dask.base import tokenize
from dask.dataframe.core import Series
from dask.dataframe.dispatch import tolist_dispatch
from dask.utils import is_cupy_type, random_state_data

def sample_percentiles(num_old, num_new, chunk_length, upsample=1.0, random_state=None):
    if False:
        for i in range(10):
            print('nop')
    'Construct percentiles for a chunk for repartitioning.\n\n    Adapt the number of total percentiles calculated based on the number\n    of current and new partitions.  Returned percentiles include equally\n    spaced percentiles between [0, 100], and random percentiles.  See\n    detailed discussion below.\n\n    Parameters\n    ----------\n    num_old: int\n        Number of partitions of the current object\n    num_new: int\n        Number of partitions of the new object\n    chunk_length: int\n        Number of rows of the partition\n    upsample : float\n        Multiplicative factor to increase the number of samples\n\n    Returns\n    -------\n    qs : numpy.ndarray of sorted percentiles between 0, 100\n\n    Constructing ordered (i.e., not hashed) partitions is hard.  Calculating\n    approximate percentiles for generic objects in an out-of-core fashion is\n    also hard.  Fortunately, partition boundaries don\'t need to be perfect\n    in order for partitioning to be effective, so we strive for a "good enough"\n    method that can scale to many partitions and is reasonably well-behaved for\n    a wide variety of scenarios.\n\n    Two similar approaches come to mind: (1) take a subsample of every\n    partition, then find the best new partitions for the combined subsamples;\n    and (2) calculate equally-spaced percentiles on every partition (a\n    relatively cheap operation), then merge the results.  We do both, but\n    instead of random samples, we use random percentiles.\n\n    If the number of partitions isn\'t changing, then the ratio of fixed\n    percentiles to random percentiles is 2 to 1.  If repartitioning goes from\n    a very high number of partitions to a very low number of partitions, then\n    we use more random percentiles, because a stochastic approach will be more\n    stable to potential correlations in the data that may cause a few equally-\n    spaced partitions to under-sample the data.\n\n    The more partitions there are, then the more total percentiles will get\n    calculated across all partitions.  Squaring the number of partitions\n    approximately doubles the number of total percentiles calculated, so\n    num_total_percentiles ~ sqrt(num_partitions).  We assume each partition\n    is approximately the same length.  This should provide adequate resolution\n    and allow the number of partitions to scale.\n\n    For numeric data, one could instead use T-Digest for floats and Q-Digest\n    for ints to calculate approximate percentiles.  Our current method works\n    for any dtype.\n    '
    random_percentage = 1 / (1 + (4 * num_new / num_old) ** 0.5)
    num_percentiles = upsample * num_new * (num_old + 22) ** 0.55 / num_old
    num_fixed = int(num_percentiles * (1 - random_percentage)) + 2
    num_random = int(num_percentiles * random_percentage) + 2
    if num_fixed + num_random + 5 >= chunk_length:
        return np.linspace(0, 100, chunk_length + 1)
    if not isinstance(random_state, np.random.RandomState):
        random_state = np.random.RandomState(random_state)
    q_fixed = np.linspace(0, 100, num_fixed)
    q_random = random_state.rand(num_random) * 100
    q_edges = [60 / (num_fixed - 1), 100 - 60 / (num_fixed - 1)]
    qs = np.concatenate([q_fixed, q_random, q_edges, [0, 100]])
    qs.sort()
    qs = 0.5 * (qs[:-1] + qs[1:])
    return qs

def tree_width(N, to_binary=False):
    if False:
        return 10
    'Generate tree width suitable for ``merge_sorted`` given N inputs\n\n    The larger N is, the more tasks are reduced in a single task.\n\n    In theory, this is designed so all tasks are of comparable effort.\n    '
    if N < 32:
        group_size = 2
    else:
        group_size = int(math.log(N))
    num_groups = N // group_size
    if to_binary or num_groups < 16:
        return 2 ** int(math.log(N / group_size, 2))
    else:
        return num_groups

def tree_groups(N, num_groups):
    if False:
        for i in range(10):
            print('nop')
    'Split an integer N into evenly sized and spaced groups.\n\n    >>> tree_groups(16, 6)\n    [3, 2, 3, 3, 2, 3]\n    '
    group_size = N // num_groups
    dx = num_groups
    dy = N - group_size * num_groups
    D = 2 * dy - dx
    rv = []
    for _ in range(num_groups):
        if D < 0:
            rv.append(group_size)
        else:
            rv.append(group_size + 1)
            D -= 2 * dx
        D += 2 * dy
    return rv

def create_merge_tree(func, keys, token, level=0):
    if False:
        while True:
            i = 10
    'Create a task tree that merges all the keys with a reduction function.\n\n    Parameters\n    ----------\n    func: callable\n        Reduction function that accepts a single list of values to reduce.\n    keys: iterable\n        Keys to reduce from the source dask graph.\n    token: object\n        Included in each key of the returned dict.\n    level: int, default 0\n        The token-level to begin with.\n\n    This creates a k-ary tree where k depends on the current level and is\n    greater the further away a node is from the root node.  This reduces the\n    total number of nodes (thereby reducing scheduler overhead), but still\n    has beneficial properties of trees.\n\n    For reasonable numbers of keys, N < 1e5, the total number of nodes in the\n    tree is roughly ``N**0.78``.  For 1e5 < N < 2e5, is it roughly ``N**0.8``.\n    '
    prev_width = len(keys)
    prev_keys = iter(keys)
    rv = {}
    while prev_width > 1:
        width = tree_width(prev_width)
        groups = tree_groups(prev_width, width)
        keys = [(token, level, i) for i in range(width)]
        for (num, key) in zip(groups, keys):
            rv[key] = (func, list(take(num, prev_keys)))
        prev_width = width
        prev_keys = iter(keys)
        level += 1
    return rv

def percentiles_to_weights(qs, vals, length):
    if False:
        print('Hello World!')
    'Weigh percentile values by length and the difference between percentiles\n\n    >>> percentiles = np.array([0., 25., 50., 90., 100.])\n    >>> values = np.array([2, 3, 5, 8, 13])\n    >>> length = 10\n    >>> percentiles_to_weights(percentiles, values, length)\n    ([2, 3, 5, 8, 13], [125.0, 250.0, 325.0, 250.0, 50.0])\n\n    The weight of the first element, ``2``, is determined by the difference\n    between the first and second percentiles, and then scaled by length:\n\n    >>> 0.5 * length * (percentiles[1] - percentiles[0])\n    125.0\n\n    The second weight uses the difference of percentiles on both sides, so\n    it will be twice the first weight if the percentiles are equally spaced:\n\n    >>> 0.5 * length * (percentiles[2] - percentiles[0])\n    250.0\n    '
    if length == 0:
        return ()
    diff = np.ediff1d(qs, 0.0, 0.0)
    weights = 0.5 * length * (diff[1:] + diff[:-1])
    try:
        return (tolist_dispatch(vals), weights.tolist())
    except TypeError:
        return (vals.tolist(), weights.tolist())

def merge_and_compress_summaries(vals_and_weights):
    if False:
        return 10
    'Merge and sort percentile summaries that are already sorted.\n\n    Each item is a tuple like ``(vals, weights)`` where vals and weights\n    are lists.  We sort both by vals.\n\n    Equal values will be combined, their weights summed together.\n    '
    vals_and_weights = [x for x in vals_and_weights if x]
    if not vals_and_weights:
        return ()
    it = merge_sorted(*[zip(x, y) for (x, y) in vals_and_weights])
    vals = []
    weights = []
    vals_append = vals.append
    weights_append = weights.append
    (val, weight) = (prev_val, prev_weight) = next(it)
    for (val, weight) in it:
        if val == prev_val:
            prev_weight += weight
        else:
            vals_append(prev_val)
            weights_append(prev_weight)
            (prev_val, prev_weight) = (val, weight)
    if val == prev_val:
        vals_append(prev_val)
        weights_append(prev_weight)
    return (vals, weights)

def process_val_weights(vals_and_weights, npartitions, dtype_info):
    if False:
        while True:
            i = 10
    "Calculate final approximate percentiles given weighted vals\n\n    ``vals_and_weights`` is assumed to be sorted.  We take a cumulative\n    sum of the weights, which makes them percentile-like (their scale is\n    [0, N] instead of [0, 100]).  Next we find the divisions to create\n    partitions of approximately equal size.\n\n    It is possible for adjacent values of the result to be the same.  Since\n    these determine the divisions of the new partitions, some partitions\n    may be empty.  This can happen if we under-sample the data, or if there\n    aren't enough unique values in the column.  Increasing ``upsample``\n    keyword argument in ``df.set_index`` may help.\n    "
    (dtype, info) = dtype_info
    if not vals_and_weights:
        try:
            return np.array(None, dtype=dtype)
        except Exception:
            return np.array(None, dtype=np.float64)
    (vals, weights) = vals_and_weights
    vals = np.array(vals)
    weights = np.array(weights)
    if len(vals) == npartitions + 1:
        rv = vals
    elif len(vals) < npartitions + 1:
        if np.issubdtype(vals.dtype, np.number) and (not isinstance(dtype, pd.CategoricalDtype)):
            q_weights = np.cumsum(weights)
            q_target = np.linspace(q_weights[0], q_weights[-1], npartitions + 1)
            rv = np.interp(q_target, q_weights, vals)
        else:
            duplicated_index = np.linspace(0, len(vals) - 1, npartitions - len(vals) + 1, dtype=int)
            duplicated_vals = vals[duplicated_index]
            rv = np.concatenate([vals, duplicated_vals])
            rv.sort()
    else:
        target_weight = weights.sum() / npartitions
        jumbo_mask = weights >= target_weight
        jumbo_vals = vals[jumbo_mask]
        trimmed_vals = vals[~jumbo_mask]
        trimmed_weights = weights[~jumbo_mask]
        trimmed_npartitions = npartitions - len(jumbo_vals)
        q_weights = np.cumsum(trimmed_weights)
        q_target = np.linspace(0, q_weights[-1], trimmed_npartitions + 1)
        left = np.searchsorted(q_weights, q_target, side='left')
        right = np.searchsorted(q_weights, q_target, side='right') - 1
        np.maximum(right, 0, right)
        lower = np.minimum(left, right)
        trimmed = trimmed_vals[lower]
        rv = np.concatenate([trimmed, jumbo_vals])
        rv.sort()
    if isinstance(dtype, pd.CategoricalDtype):
        rv = pd.Categorical.from_codes(rv, info[0], info[1])
    elif isinstance(dtype, pd.DatetimeTZDtype):
        rv = pd.DatetimeIndex(rv).tz_localize(dtype.tz)
    elif 'datetime64' in str(dtype):
        rv = pd.DatetimeIndex(rv, dtype=dtype)
    elif rv.dtype != dtype:
        if is_integer_dtype(dtype) and pd.api.types.is_float_dtype(rv.dtype):
            rv = np.floor(rv)
        rv = pd.array(rv, dtype=dtype)
    return rv

def percentiles_summary(df, num_old, num_new, upsample, state):
    if False:
        while True:
            i = 10
    'Summarize data using percentiles and derived weights.\n\n    These summaries can be merged, compressed, and converted back into\n    approximate percentiles.\n\n    Parameters\n    ----------\n    df: pandas.Series\n        Data to summarize\n    num_old: int\n        Number of partitions of the current object\n    num_new: int\n        Number of partitions of the new object\n    upsample: float\n        Scale factor to increase the number of percentiles calculated in\n        each partition.  Use to improve accuracy.\n    '
    from dask.array.dispatch import percentile_lookup as _percentile
    from dask.array.utils import array_safe
    length = len(df)
    if length == 0:
        return ()
    random_state = np.random.RandomState(state)
    qs = sample_percentiles(num_old, num_new, length, upsample, random_state)
    data = df
    interpolation = 'linear'
    if isinstance(data.dtype, pd.CategoricalDtype):
        data = data.cat.codes
        interpolation = 'nearest'
    elif is_datetime64_dtype(data.dtype) or is_integer_dtype(data.dtype):
        interpolation = 'nearest'
    try:
        vals = data.quantile(q=qs / 100, interpolation=interpolation).values
    except (TypeError, NotImplementedError):
        try:
            (vals, _) = _percentile(array_safe(data, like=data.values), qs, interpolation)
        except (TypeError, NotImplementedError):
            interpolation = 'nearest'
            vals = data.to_frame().quantile(q=qs / 100, interpolation=interpolation, numeric_only=False, method='table').iloc[:, 0]
    if is_cupy_type(data) and interpolation == 'linear' and np.issubdtype(data.dtype, np.integer):
        vals = np.round(vals).astype(data.dtype)
        if qs[0] == 0:
            vals[0] = data.min()
    vals_and_weights = percentiles_to_weights(qs, vals, length)
    return vals_and_weights

def dtype_info(df):
    if False:
        print('Hello World!')
    info = None
    if isinstance(df.dtype, pd.CategoricalDtype):
        data = df.values
        info = (data.categories, data.ordered)
    return (df.dtype, info)

def partition_quantiles(df, npartitions, upsample=1.0, random_state=None):
    if False:
        i = 10
        return i + 15
    'Approximate quantiles of Series used for repartitioning'
    assert isinstance(df, Series)
    return_type = Series
    qs = np.linspace(0, 1, npartitions + 1)
    token = tokenize(df, qs, upsample)
    if random_state is None:
        random_state = int(token, 16) % np.iinfo(np.int32).max
    state_data = random_state_data(df.npartitions, random_state)
    df_keys = df.__dask_keys__()
    name0 = 're-quantiles-0-' + token
    dtype_dsk = {(name0, 0): (dtype_info, df_keys[0])}
    name1 = 're-quantiles-1-' + token
    val_dsk = {(name1, i): (percentiles_summary, key, df.npartitions, npartitions, upsample, state) for (i, (state, key)) in enumerate(zip(state_data, df_keys))}
    name2 = 're-quantiles-2-' + token
    merge_dsk = create_merge_tree(merge_and_compress_summaries, sorted(val_dsk), name2)
    if not merge_dsk:
        merge_dsk = {(name2, 0, 0): (merge_and_compress_summaries, [list(val_dsk)[0]])}
    merged_key = max(merge_dsk)
    name3 = 're-quantiles-3-' + token
    last_dsk = {(name3, 0): (pd.Series, (process_val_weights, merged_key, npartitions, (name0, 0)), qs, None, df.name)}
    dsk = merge(df.dask, dtype_dsk, val_dsk, merge_dsk, last_dsk)
    new_divisions = [0.0, 1.0]
    return return_type(dsk, name3, df._meta, new_divisions)