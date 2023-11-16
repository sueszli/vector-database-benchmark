from __future__ import annotations
import bisect
import collections
import functools
from river import sketch
from river.sketch.histogram import Bin
from ..utils import BranchFactory
from .base import Splitter

class HistogramSplitter(Splitter):
    """Numeric attribute observer for classification tasks that discretizes features
    using histograms.


    Parameters
    ----------
    n_bins
        The maximum number of bins in the histogram.
    n_splits
        The number of split points to evaluate when querying for the best split
        candidate.
    """

    def __init__(self, n_bins: int=256, n_splits: int=32):
        if False:
            return 10
        super().__init__()
        self.n_bins = n_bins
        self.n_splits = n_splits
        self.hists: collections.defaultdict = collections.defaultdict(functools.partial(sketch.Histogram, max_bins=self.n_bins))

    def update(self, att_val, target_val, sample_weight):
        if False:
            for i in range(10):
                print('nop')
        for _ in range(int(sample_weight)):
            self.hists[target_val].update(att_val)

    def cond_proba(self, att_val, target_val):
        if False:
            for i in range(10):
                print('nop')
        if target_val not in self.hists:
            return 0.0
        total_weight = self.hists[target_val].n
        if not total_weight > 0:
            return 0.0
        i = bisect.bisect(self.hists[target_val], Bin(att_val, att_val, 1))
        if i < len(self.hists[target_val]):
            b = self.hists[target_val][i]
        else:
            b = self.hists[target_val][-1]
        if b.left == b.right:
            return b.count / total_weight
        else:
            return b.count * (att_val - b.left) / (b.right - b.left) / total_weight

    def best_evaluated_split_suggestion(self, criterion, pre_split_dist, att_idx, binary_only):
        if False:
            print('Hello World!')
        best_suggestion = BranchFactory()
        low = min((h[0].right for h in self.hists.values()))
        high = min((h[-1].right for h in self.hists.values()))
        if low >= high:
            return best_suggestion
        n_thresholds = min(self.n_splits, max(map(len, self.hists.values())) - 1)
        thresholds = list(decimal_range(start=low, stop=high, num=n_thresholds))
        cdfs = {y: hist.iter_cdf(thresholds) for (y, hist) in self.hists.items()}
        total_weight = sum(pre_split_dist.values())
        for at in thresholds:
            l_dist = {}
            r_dist = {}
            for y in pre_split_dist:
                if y in cdfs:
                    p_xy = next(cdfs[y])
                    p_y = pre_split_dist[y] / total_weight
                    l_dist[y] = total_weight * p_y * p_xy
                    r_dist[y] = total_weight * p_y * (1 - p_xy)
            post_split_dist = [l_dist, r_dist]
            merit = criterion.merit_of_split(pre_split_dist, post_split_dist)
            if merit > best_suggestion.merit:
                best_suggestion = BranchFactory(merit, att_idx, at, post_split_dist)
        return best_suggestion

def decimal_range(start, stop, num):
    if False:
        return 10
    '\n    Example\n    -------\n    >>> for x in decimal_range(0, 1, 4):\n    ...     print(x)\n    0.2\n    0.4\n    0.6\n    0.8\n    '
    step = (stop - start) / (num + 1)
    for _ in range(num):
        start += step
        yield start