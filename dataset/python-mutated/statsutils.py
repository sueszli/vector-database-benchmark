"""``statsutils`` provides tools aimed primarily at descriptive
statistics for data analysis, such as :func:`mean` (average),
:func:`median`, :func:`variance`, and many others,

The :class:`Stats` type provides all the main functionality of the
``statsutils`` module. A :class:`Stats` object wraps a given dataset,
providing all statistical measures as property attributes. These
attributes cache their results, which allows efficient computation of
multiple measures, as many measures rely on other measures. For
example, relative standard deviation (:attr:`Stats.rel_std_dev`)
relies on both the mean and standard deviation. The Stats object
caches those results so no rework is done.

The :class:`Stats` type's attributes have module-level counterparts for
convenience when the computation reuse advantages do not apply.

>>> stats = Stats(range(42))
>>> stats.mean
20.5
>>> mean(range(42))
20.5

Statistics is a large field, and ``statsutils`` is focused on a few
basic techniques that are useful in software. The following is a brief
introduction to those techniques. For a more in-depth introduction,
`Statistics for Software
<https://www.paypal-engineering.com/2016/04/11/statistics-for-software/>`_,
an article I wrote on the topic. It introduces key terminology vital
to effective usage of statistics.

Statistical moments
-------------------

Python programmers are probably familiar with the concept of the
*mean* or *average*, which gives a rough quantitiative middle value by
which a sample can be can be generalized. However, the mean is just
the first of four `moment`_-based measures by which a sample or
distribution can be measured.

The four `Standardized moments`_ are:

  1. `Mean`_ - :func:`mean` - theoretical middle value
  2. `Variance`_ - :func:`variance` - width of value dispersion
  3. `Skewness`_ - :func:`skewness` - symmetry of distribution
  4. `Kurtosis`_ - :func:`kurtosis` - "peakiness" or "long-tailed"-ness

For more information check out `the Moment article on Wikipedia`_.

.. _moment: https://en.wikipedia.org/wiki/Moment_(mathematics)
.. _Standardized moments: https://en.wikipedia.org/wiki/Standardized_moment
.. _Mean: https://en.wikipedia.org/wiki/Mean
.. _Variance: https://en.wikipedia.org/wiki/Variance
.. _Skewness: https://en.wikipedia.org/wiki/Skewness
.. _Kurtosis: https://en.wikipedia.org/wiki/Kurtosis
.. _the Moment article on Wikipedia: https://en.wikipedia.org/wiki/Moment_(mathematics)

Keep in mind that while these moments can give a bit more insight into
the shape and distribution of data, they do not guarantee a complete
picture. Wildly different datasets can have the same values for all
four moments, so generalize wisely.

Robust statistics
-----------------

Moment-based statistics are notorious for being easily skewed by
outliers. The whole field of robust statistics aims to mitigate this
dilemma. ``statsutils`` also includes several robust statistical methods:

  * `Median`_ - The middle value of a sorted dataset
  * `Trimean`_ - Another robust measure of the data's central tendency
  * `Median Absolute Deviation`_ (MAD) - A robust measure of
    variability, a natural counterpart to :func:`variance`.
  * `Trimming`_ - Reducing a dataset to only the middle majority of
    data is a simple way of making other estimators more robust.

.. _Median: https://en.wikipedia.org/wiki/Median
.. _Trimean: https://en.wikipedia.org/wiki/Trimean
.. _Median Absolute Deviation: https://en.wikipedia.org/wiki/Median_absolute_deviation
.. _Trimming: https://en.wikipedia.org/wiki/Trimmed_estimator


Online and Offline Statistics
-----------------------------

Unrelated to computer networking, `online`_ statistics involve
calculating statistics in a `streaming`_ fashion, without all the data
being available. The :class:`Stats` type is meant for the more
traditional offline statistics when all the data is available. For
pure-Python online statistics accumulators, look at the `Lithoxyl`_
system instrumentation package.

.. _Online: https://en.wikipedia.org/wiki/Online_algorithm
.. _streaming: https://en.wikipedia.org/wiki/Streaming_algorithm
.. _Lithoxyl: https://github.com/mahmoud/lithoxyl

"""
from __future__ import print_function
import bisect
from math import floor, ceil

class _StatsProperty(object):

    def __init__(self, name, func):
        if False:
            while True:
                i = 10
        self.name = name
        self.func = func
        self.internal_name = '_' + name
        doc = func.__doc__ or ''
        (pre_doctest_doc, _, _) = doc.partition('>>>')
        self.__doc__ = pre_doctest_doc

    def __get__(self, obj, objtype=None):
        if False:
            print('Hello World!')
        if obj is None:
            return self
        if not obj.data:
            return obj.default
        try:
            return getattr(obj, self.internal_name)
        except AttributeError:
            setattr(obj, self.internal_name, self.func(obj))
            return getattr(obj, self.internal_name)

class Stats(object):
    """The ``Stats`` type is used to represent a group of unordered
    statistical datapoints for calculations such as mean, median, and
    variance.

    Args:

        data (list): List or other iterable containing numeric values.
        default (float): A value to be returned when a given
            statistical measure is not defined. 0.0 by default, but
            ``float('nan')`` is appropriate for stricter applications.
        use_copy (bool): By default Stats objects copy the initial
            data into a new list to avoid issues with
            modifications. Pass ``False`` to disable this behavior.
        is_sorted (bool): Presorted data can skip an extra sorting
            step for a little speed boost. Defaults to False.

    """

    def __init__(self, data, default=0.0, use_copy=True, is_sorted=False):
        if False:
            print('Hello World!')
        self._use_copy = use_copy
        self._is_sorted = is_sorted
        if use_copy:
            self.data = list(data)
        else:
            self.data = data
        self.default = default
        cls = self.__class__
        self._prop_attr_names = [a for a in dir(self) if isinstance(getattr(cls, a, None), _StatsProperty)]
        self._pearson_precision = 0

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.data)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self.data)

    def _get_sorted_data(self):
        if False:
            while True:
                i = 10
        "When using a copy of the data, it's better to have that copy be\n        sorted, but we do it lazily using this method, in case no\n        sorted measures are used. I.e., if median is never called,\n        sorting would be a waste.\n\n        When not using a copy, it's presumed that all optimizations\n        are on the user.\n        "
        if not self._use_copy:
            return sorted(self.data)
        elif not self._is_sorted:
            self.data.sort()
        return self.data

    def clear_cache(self):
        if False:
            for i in range(10):
                print('nop')
        '``Stats`` objects automatically cache intermediary calculations\n        that can be reused. For instance, accessing the ``std_dev``\n        attribute after the ``variance`` attribute will be\n        significantly faster for medium-to-large datasets.\n\n        If you modify the object by adding additional data points,\n        call this function to have the cached statistics recomputed.\n\n        '
        for attr_name in self._prop_attr_names:
            attr_name = getattr(self.__class__, attr_name).internal_name
            if not hasattr(self, attr_name):
                continue
            delattr(self, attr_name)
        return

    def _calc_count(self):
        if False:
            for i in range(10):
                print('nop')
        'The number of items in this Stats object. Returns the same as\n        :func:`len` on a Stats object, but provided for pandas terminology\n        parallelism.\n\n        >>> Stats(range(20)).count\n        20\n        '
        return len(self.data)
    count = _StatsProperty('count', _calc_count)

    def _calc_mean(self):
        if False:
            i = 10
            return i + 15
        '\n        The arithmetic mean, or "average". Sum of the values divided by\n        the number of values.\n\n        >>> mean(range(20))\n        9.5\n        >>> mean(list(range(19)) + [949])  # 949 is an arbitrary outlier\n        56.0\n        '
        return sum(self.data, 0.0) / len(self.data)
    mean = _StatsProperty('mean', _calc_mean)

    def _calc_max(self):
        if False:
            return 10
        '\n        The maximum value present in the data.\n\n        >>> Stats([2, 1, 3]).max\n        3\n        '
        if self._is_sorted:
            return self.data[-1]
        return max(self.data)
    max = _StatsProperty('max', _calc_max)

    def _calc_min(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The minimum value present in the data.\n\n        >>> Stats([2, 1, 3]).min\n        1\n        '
        if self._is_sorted:
            return self.data[0]
        return min(self.data)
    min = _StatsProperty('min', _calc_min)

    def _calc_median(self):
        if False:
            return 10
        "\n        The median is either the middle value or the average of the two\n        middle values of a sample. Compared to the mean, it's generally\n        more resilient to the presence of outliers in the sample.\n\n        >>> median([2, 1, 3])\n        2\n        >>> median(range(97))\n        48\n        >>> median(list(range(96)) + [1066])  # 1066 is an arbitrary outlier\n        48\n        "
        return self._get_quantile(self._get_sorted_data(), 0.5)
    median = _StatsProperty('median', _calc_median)

    def _calc_iqr(self):
        if False:
            for i in range(10):
                print('nop')
        'Inter-quartile range (IQR) is the difference between the 75th\n        percentile and 25th percentile. IQR is a robust measure of\n        dispersion, like standard deviation, but safer to compare\n        between datasets, as it is less influenced by outliers.\n\n        >>> iqr([1, 2, 3, 4, 5])\n        2\n        >>> iqr(range(1001))\n        500\n        '
        return self.get_quantile(0.75) - self.get_quantile(0.25)
    iqr = _StatsProperty('iqr', _calc_iqr)

    def _calc_trimean(self):
        if False:
            return 10
        'The trimean is a robust measure of central tendency, like the\n        median, that takes the weighted average of the median and the\n        upper and lower quartiles.\n\n        >>> trimean([2, 1, 3])\n        2.0\n        >>> trimean(range(97))\n        48.0\n        >>> trimean(list(range(96)) + [1066])  # 1066 is an arbitrary outlier\n        48.0\n\n        '
        sorted_data = self._get_sorted_data()
        gq = lambda q: self._get_quantile(sorted_data, q)
        return (gq(0.25) + 2 * gq(0.5) + gq(0.75)) / 4.0
    trimean = _StatsProperty('trimean', _calc_trimean)

    def _calc_variance(self):
        if False:
            return 10
        '        Variance is the average of the squares of the difference between\n        each value and the mean.\n\n        >>> variance(range(97))\n        784.0\n        '
        global mean
        return mean(self._get_pow_diffs(2))
    variance = _StatsProperty('variance', _calc_variance)

    def _calc_std_dev(self):
        if False:
            while True:
                i = 10
        '        Standard deviation. Square root of the variance.\n\n        >>> std_dev(range(97))\n        28.0\n        '
        return self.variance ** 0.5
    std_dev = _StatsProperty('std_dev', _calc_std_dev)

    def _calc_median_abs_dev(self):
        if False:
            return 10
        '        Median Absolute Deviation is a robust measure of statistical\n        dispersion: http://en.wikipedia.org/wiki/Median_absolute_deviation\n\n        >>> median_abs_dev(range(97))\n        24.0\n        '
        global median
        sorted_vals = sorted(self.data)
        x = float(median(sorted_vals))
        return median([abs(x - v) for v in sorted_vals])
    median_abs_dev = _StatsProperty('median_abs_dev', _calc_median_abs_dev)
    mad = median_abs_dev

    def _calc_rel_std_dev(self):
        if False:
            return 10
        "        Standard deviation divided by the absolute value of the average.\n\n        http://en.wikipedia.org/wiki/Relative_standard_deviation\n\n        >>> print('%1.3f' % rel_std_dev(range(97)))\n        0.583\n        "
        abs_mean = abs(self.mean)
        if abs_mean:
            return self.std_dev / abs_mean
        else:
            return self.default
    rel_std_dev = _StatsProperty('rel_std_dev', _calc_rel_std_dev)

    def _calc_skewness(self):
        if False:
            i = 10
            return i + 15
        '        Indicates the asymmetry of a curve. Positive values mean the bulk\n        of the values are on the left side of the average and vice versa.\n\n        http://en.wikipedia.org/wiki/Skewness\n\n        See the module docstring for more about statistical moments.\n\n        >>> skewness(range(97))  # symmetrical around 48.0\n        0.0\n        >>> left_skewed = skewness(list(range(97)) + list(range(10)))\n        >>> right_skewed = skewness(list(range(97)) + list(range(87, 97)))\n        >>> round(left_skewed, 3), round(right_skewed, 3)\n        (0.114, -0.114)\n        '
        (data, s_dev) = (self.data, self.std_dev)
        if len(data) > 1 and s_dev > 0:
            return sum(self._get_pow_diffs(3)) / float((len(data) - 1) * s_dev ** 3)
        else:
            return self.default
    skewness = _StatsProperty('skewness', _calc_skewness)

    def _calc_kurtosis(self):
        if False:
            print('Hello World!')
        '        Indicates how much data is in the tails of the distribution. The\n        result is always positive, with the normal "bell-curve"\n        distribution having a kurtosis of 3.\n\n        http://en.wikipedia.org/wiki/Kurtosis\n\n        See the module docstring for more about statistical moments.\n\n        >>> kurtosis(range(9))\n        1.99125\n\n        With a kurtosis of 1.99125, [0, 1, 2, 3, 4, 5, 6, 7, 8] is more\n        centrally distributed than the normal curve.\n        '
        (data, s_dev) = (self.data, self.std_dev)
        if len(data) > 1 and s_dev > 0:
            return sum(self._get_pow_diffs(4)) / float((len(data) - 1) * s_dev ** 4)
        else:
            return 0.0
    kurtosis = _StatsProperty('kurtosis', _calc_kurtosis)

    def _calc_pearson_type(self):
        if False:
            print('Hello World!')
        precision = self._pearson_precision
        skewness = self.skewness
        kurtosis = self.kurtosis
        beta1 = skewness ** 2.0
        beta2 = kurtosis * 1.0
        c0 = 4 * beta2 - 3 * beta1
        c1 = skewness * (beta2 + 3)
        c2 = 2 * beta2 - 3 * beta1 - 6
        if round(c1, precision) == 0:
            if round(beta2, precision) == 3:
                return 0
            elif beta2 < 3:
                return 2
            elif beta2 > 3:
                return 7
        elif round(c2, precision) == 0:
            return 3
        else:
            k = c1 ** 2 / (4 * c0 * c2)
            if k < 0:
                return 1
        raise RuntimeError('missed a spot')
    pearson_type = _StatsProperty('pearson_type', _calc_pearson_type)

    @staticmethod
    def _get_quantile(sorted_data, q):
        if False:
            for i in range(10):
                print('nop')
        (data, n) = (sorted_data, len(sorted_data))
        idx = q / 1.0 * (n - 1)
        (idx_f, idx_c) = (int(floor(idx)), int(ceil(idx)))
        if idx_f == idx_c:
            return data[idx_f]
        return data[idx_f] * (idx_c - idx) + data[idx_c] * (idx - idx_f)

    def get_quantile(self, q):
        if False:
            return 10
        'Get a quantile from the dataset. Quantiles are floating point\n        values between ``0.0`` and ``1.0``, with ``0.0`` representing\n        the minimum value in the dataset and ``1.0`` representing the\n        maximum. ``0.5`` represents the median:\n\n        >>> Stats(range(100)).get_quantile(0.5)\n        49.5\n        '
        q = float(q)
        if not 0.0 <= q <= 1.0:
            raise ValueError('expected q between 0.0 and 1.0, not %r' % q)
        elif not self.data:
            return self.default
        return self._get_quantile(self._get_sorted_data(), q)

    def get_zscore(self, value):
        if False:
            print('Hello World!')
        "Get the z-score for *value* in the group. If the standard deviation\n        is 0, 0 inf or -inf will be returned to indicate whether the value is\n        equal to, greater than or below the group's mean.\n        "
        mean = self.mean
        if self.std_dev == 0:
            if value == mean:
                return 0
            if value > mean:
                return float('inf')
            if value < mean:
                return float('-inf')
        return (float(value) - mean) / self.std_dev

    def trim_relative(self, amount=0.15):
        if False:
            print('Hello World!')
        'A utility function used to cut a proportion of values off each end\n        of a list of values. This has the effect of limiting the\n        effect of outliers.\n\n        Args:\n            amount (float): A value between 0.0 and 0.5 to trim off of\n                each side of the data.\n\n        .. note:\n\n            This operation modifies the data in-place. It does not\n            make or return a copy.\n\n        '
        trim = float(amount)
        if not 0.0 <= trim < 0.5:
            raise ValueError('expected amount between 0.0 and 0.5, not %r' % trim)
        size = len(self.data)
        size_diff = int(size * trim)
        if size_diff == 0.0:
            return
        self.data = self._get_sorted_data()[size_diff:-size_diff]
        self.clear_cache()

    def _get_pow_diffs(self, power):
        if False:
            return 10
        '\n        A utility function used for calculating statistical moments.\n        '
        m = self.mean
        return [(v - m) ** power for v in self.data]

    def _get_bin_bounds(self, count=None, with_max=False):
        if False:
            for i in range(10):
                print('nop')
        if not self.data:
            return [0.0]
        data = self.data
        (len_data, min_data, max_data) = (len(data), min(data), max(data))
        if len_data < 4:
            if not count:
                count = len_data
            dx = (max_data - min_data) / float(count)
            bins = [min_data + dx * i for i in range(count)]
        elif count is None:
            (q25, q75) = (self.get_quantile(0.25), self.get_quantile(0.75))
            dx = 2 * (q75 - q25) / len_data ** (1 / 3.0)
            bin_count = max(1, int(ceil((max_data - min_data) / dx)))
            bins = [min_data + dx * i for i in range(bin_count + 1)]
            bins = [b for b in bins if b < max_data]
        else:
            dx = (max_data - min_data) / float(count)
            bins = [min_data + dx * i for i in range(count)]
        if with_max:
            bins.append(float(max_data))
        return bins

    def get_histogram_counts(self, bins=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        "Produces a list of ``(bin, count)`` pairs comprising a histogram of\n        the Stats object's data, using fixed-width bins. See\n        :meth:`Stats.format_histogram` for more details.\n\n        Args:\n            bins (int): maximum number of bins, or list of\n                floating-point bin boundaries. Defaults to the output of\n                Freedman's algorithm.\n            bin_digits (int): Number of digits used to round down the\n                bin boundaries. Defaults to 1.\n\n        The output of this method can be stored and/or modified, and\n        then passed to :func:`statsutils.format_histogram_counts` to\n        achieve the same text formatting as the\n        :meth:`~Stats.format_histogram` method. This can be useful for\n        snapshotting over time.\n        "
        bin_digits = int(kw.pop('bin_digits', 1))
        if kw:
            raise TypeError('unexpected keyword arguments: %r' % kw.keys())
        if not bins:
            bins = self._get_bin_bounds()
        else:
            try:
                bin_count = int(bins)
            except TypeError:
                try:
                    bins = [float(x) for x in bins]
                except Exception:
                    raise ValueError('bins expected integer bin count or list of float bin boundaries, not %r' % bins)
                if self.min < bins[0]:
                    bins = [self.min] + bins
            else:
                bins = self._get_bin_bounds(bin_count)
        round_factor = 10.0 ** bin_digits
        bins = [floor(b * round_factor) / round_factor for b in bins]
        bins = sorted(set(bins))
        idxs = [bisect.bisect(bins, d) - 1 for d in self.data]
        count_map = {}
        for idx in idxs:
            try:
                count_map[idx] += 1
            except KeyError:
                count_map[idx] = 1
        bin_counts = [(b, count_map.get(i, 0)) for (i, b) in enumerate(bins)]
        return bin_counts

    def format_histogram(self, bins=None, **kw):
        if False:
            while True:
                i = 10
        'Produces a textual histogram of the data, using fixed-width bins,\n        allowing for simple visualization, even in console environments.\n\n        >>> data = list(range(20)) + list(range(5, 15)) + [10]\n        >>> print(Stats(data).format_histogram(width=30))\n         0.0:  5 #########\n         4.4:  8 ###############\n         8.9: 11 ####################\n        13.3:  5 #########\n        17.8:  2 ####\n\n        In this histogram, five values are between 0.0 and 4.4, eight\n        are between 4.4 and 8.9, and two values lie between 17.8 and\n        the max.\n\n        You can specify the number of bins, or provide a list of\n        bin boundaries themselves. If no bins are provided, as in the\n        example above, `Freedman\'s algorithm`_ for bin selection is\n        used.\n\n        Args:\n            bins (int): Maximum number of bins for the\n                histogram. Also accepts a list of floating-point\n                bin boundaries. If the minimum boundary is still\n                greater than the minimum value in the data, that\n                boundary will be implicitly added. Defaults to the bin\n                boundaries returned by `Freedman\'s algorithm`_.\n            bin_digits (int): Number of digits to round each bin\n                to. Note that bins are always rounded down to avoid\n                clipping any data. Defaults to 1.\n            width (int): integer number of columns in the longest line\n               in the histogram. Defaults to console width on Python\n               3.3+, or 80 if that is not available.\n            format_bin (callable): Called on each bin to create a\n               label for the final output. Use this function to add\n               units, such as "ms" for milliseconds.\n\n        Should you want something more programmatically reusable, see\n        the :meth:`~Stats.get_histogram_counts` method, the output of\n        is used by format_histogram. The :meth:`~Stats.describe`\n        method is another useful summarization method, albeit less\n        visual.\n\n        .. _Freedman\'s algorithm: https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule\n        '
        width = kw.pop('width', None)
        format_bin = kw.pop('format_bin', None)
        bin_counts = self.get_histogram_counts(bins=bins, **kw)
        return format_histogram_counts(bin_counts, width=width, format_bin=format_bin)

    def describe(self, quantiles=None, format=None):
        if False:
            while True:
                i = 10
        'Provides standard summary statistics for the data in the Stats\n        object, in one of several convenient formats.\n\n        Args:\n            quantiles (list): A list of numeric values to use as\n                quantiles in the resulting summary. All values must be\n                0.0-1.0, with 0.5 representing the median. Defaults to\n                ``[0.25, 0.5, 0.75]``, representing the standard\n                quartiles.\n            format (str): Controls the return type of the function,\n                with one of three valid values: ``"dict"`` gives back\n                a :class:`dict` with the appropriate keys and\n                values. ``"list"`` is a list of key-value pairs in an\n                order suitable to pass to an OrderedDict or HTML\n                table. ``"text"`` converts the values to text suitable\n                for printing, as seen below.\n\n        Here is the information returned by a default ``describe``, as\n        presented in the ``"text"`` format:\n\n        >>> stats = Stats(range(1, 8))\n        >>> print(stats.describe(format=\'text\'))\n        count:    7\n        mean:     4.0\n        std_dev:  2.0\n        mad:      2.0\n        min:      1\n        0.25:     2.5\n        0.5:      4\n        0.75:     5.5\n        max:      7\n\n        For more advanced descriptive statistics, check out my blog\n        post on the topic `Statistics for Software\n        <https://www.paypal-engineering.com/2016/04/11/statistics-for-software/>`_.\n\n        '
        if format is None:
            format = 'dict'
        elif format not in ('dict', 'list', 'text'):
            raise ValueError('invalid format for describe, expected one of "dict"/"list"/"text", not %r' % format)
        quantiles = quantiles or [0.25, 0.5, 0.75]
        q_items = []
        for q in quantiles:
            q_val = self.get_quantile(q)
            q_items.append((str(q), q_val))
        items = [('count', self.count), ('mean', self.mean), ('std_dev', self.std_dev), ('mad', self.mad), ('min', self.min)]
        items.extend(q_items)
        items.append(('max', self.max))
        if format == 'dict':
            ret = dict(items)
        elif format == 'list':
            ret = items
        elif format == 'text':
            ret = '\n'.join(['%s%s' % ((label + ':').ljust(10), val) for (label, val) in items])
        return ret

def describe(data, quantiles=None, format=None):
    if False:
        i = 10
        return i + 15
    "A convenience function to get standard summary statistics useful\n    for describing most data. See :meth:`Stats.describe` for more\n    details.\n\n    >>> print(describe(range(7), format='text'))\n    count:    7\n    mean:     3.0\n    std_dev:  2.0\n    mad:      2.0\n    min:      0\n    0.25:     1.5\n    0.5:      3\n    0.75:     4.5\n    max:      6\n\n    See :meth:`Stats.format_histogram` for another very useful\n    summarization that uses textual visualization.\n    "
    return Stats(data).describe(quantiles=quantiles, format=format)

def _get_conv_func(attr_name):
    if False:
        i = 10
        return i + 15

    def stats_helper(data, default=0.0):
        if False:
            while True:
                i = 10
        return getattr(Stats(data, default=default, use_copy=False), attr_name)
    return stats_helper
for (attr_name, attr) in list(Stats.__dict__.items()):
    if isinstance(attr, _StatsProperty):
        if attr_name in ('max', 'min', 'count'):
            continue
        if attr_name in ('mad',):
            continue
        func = _get_conv_func(attr_name)
        func.__doc__ = attr.func.__doc__
        globals()[attr_name] = func
        delattr(Stats, '_calc_' + attr_name)
del attr
del attr_name
del func

def format_histogram_counts(bin_counts, width=None, format_bin=None):
    if False:
        while True:
            i = 10
    'The formatting logic behind :meth:`Stats.format_histogram`, which\n    takes the output of :meth:`Stats.get_histogram_counts`, and passes\n    them to this function.\n\n    Args:\n        bin_counts (list): A list of bin values to counts.\n        width (int): Number of character columns in the text output,\n            defaults to 80 or console width in Python 3.3+.\n        format_bin (callable): Used to convert bin values into string\n            labels.\n    '
    lines = []
    if not format_bin:
        format_bin = lambda v: v
    if not width:
        try:
            import shutil
            width = shutil.get_terminal_size()[0]
        except Exception:
            width = 80
    bins = [b for (b, _) in bin_counts]
    count_max = max([count for (_, count) in bin_counts])
    count_cols = len(str(count_max))
    labels = ['%s' % format_bin(b) for b in bins]
    label_cols = max([len(l) for l in labels])
    tmp_line = '%s: %s #' % ('x' * label_cols, count_max)
    bar_cols = max(width - len(tmp_line), 3)
    line_k = float(bar_cols) / count_max
    tmpl = '{label:>{label_cols}}: {count:>{count_cols}} {bar}'
    for (label, (bin_val, count)) in zip(labels, bin_counts):
        bar_len = int(round(count * line_k))
        bar = '#' * bar_len or '|'
        line = tmpl.format(label=label, label_cols=label_cols, count=count, count_cols=count_cols, bar=bar)
        lines.append(line)
    return '\n'.join(lines)