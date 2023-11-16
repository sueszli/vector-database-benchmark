"""This file contains code for use with "Think Stats" and
"Think Bayes", both by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""
from __future__ import print_function, division
'This file contains class definitions for:\n\nHist: represents a histogram (map from values to integer frequencies).\n\nPmf: represents a probability mass function (map from values to probs).\n\n_DictWrapper: private parent class for Hist and Pmf.\n\nCdf: represents a discrete cumulative distribution function\n\nPdf: represents a continuous probability density function\n\n'
import bisect
import copy
import logging
import math
import random
import re
from collections import Counter
from operator import itemgetter
import thinkplot
import numpy as np
import pandas
import scipy
from scipy import stats
from scipy import special
from scipy import ndimage
from io import open
ROOT2 = math.sqrt(2)

def RandomSeed(x):
    if False:
        for i in range(10):
            print('nop')
    'Initialize the random and np.random generators.\n\n    x: int seed\n    '
    random.seed(x)
    np.random.seed(x)

def Odds(p):
    if False:
        i = 10
        return i + 15
    "Computes odds for a given probability.\n\n    Example: p=0.75 means 75 for and 25 against, or 3:1 odds in favor.\n\n    Note: when p=1, the formula for odds divides by zero, which is\n    normally undefined.  But I think it is reasonable to define Odds(1)\n    to be infinity, so that's what this function does.\n\n    p: float 0-1\n\n    Returns: float odds\n    "
    if p == 1:
        return float('inf')
    return p / (1 - p)

def Probability(o):
    if False:
        while True:
            i = 10
    'Computes the probability corresponding to given odds.\n\n    Example: o=2 means 2:1 odds in favor, or 2/3 probability\n\n    o: float odds, strictly positive\n\n    Returns: float probability\n    '
    return o / (o + 1)

def Probability2(yes, no):
    if False:
        return 10
    'Computes the probability corresponding to given odds.\n\n    Example: yes=2, no=1 means 2:1 odds in favor, or 2/3 probability.\n    \n    yes, no: int or float odds in favor\n    '
    return yes / (yes + no)

class Interpolator(object):
    """Represents a mapping between sorted sequences; performs linear interp.

    Attributes:
        xs: sorted list
        ys: sorted list
    """

    def __init__(self, xs, ys):
        if False:
            for i in range(10):
                print('nop')
        self.xs = xs
        self.ys = ys

    def Lookup(self, x):
        if False:
            print('Hello World!')
        'Looks up x and returns the corresponding value of y.'
        return self._Bisect(x, self.xs, self.ys)

    def Reverse(self, y):
        if False:
            return 10
        'Looks up y and returns the corresponding value of x.'
        return self._Bisect(y, self.ys, self.xs)

    def _Bisect(self, x, xs, ys):
        if False:
            i = 10
            return i + 15
        'Helper function.'
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        i = bisect.bisect(xs, x)
        frac = 1.0 * (x - xs[i - 1]) / (xs[i] - xs[i - 1])
        y = ys[i - 1] + frac * 1.0 * (ys[i] - ys[i - 1])
        return y

class _DictWrapper(object):
    """An object that contains a dictionary."""

    def __init__(self, obj=None, label=None):
        if False:
            return 10
        'Initializes the distribution.\n\n        obj: Hist, Pmf, Cdf, Pdf, dict, pandas Series, list of pairs\n        label: string label\n        '
        self.label = label if label is not None else '_nolegend_'
        self.d = {}
        self.log = False
        if obj is None:
            return
        if isinstance(obj, (_DictWrapper, Cdf, Pdf)):
            self.label = label if label is not None else obj.label
        if isinstance(obj, dict):
            self.d.update(obj.items())
        elif isinstance(obj, (_DictWrapper, Cdf, Pdf)):
            self.d.update(obj.Items())
        elif isinstance(obj, pandas.Series):
            self.d.update(obj.value_counts().iteritems())
        else:
            self.d.update(Counter(obj))
        if len(self) > 0 and isinstance(self, Pmf):
            self.Normalize()

    def __hash__(self):
        if False:
            for i in range(10):
                print('nop')
        return id(self)

    def __str__(self):
        if False:
            print('Hello World!')
        cls = self.__class__.__name__
        return '%s(%s)' % (cls, str(self.d))
    __repr__ = __str__

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self.d == other.d

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self.d)

    def __iter__(self):
        if False:
            return 10
        return iter(self.d)

    def iterkeys(self):
        if False:
            i = 10
            return i + 15
        'Returns an iterator over keys.'
        return iter(self.d)

    def __contains__(self, value):
        if False:
            while True:
                i = 10
        return value in self.d

    def __getitem__(self, value):
        if False:
            while True:
                i = 10
        return self.d.get(value, 0)

    def __setitem__(self, value, prob):
        if False:
            print('Hello World!')
        self.d[value] = prob

    def __delitem__(self, value):
        if False:
            i = 10
            return i + 15
        del self.d[value]

    def Copy(self, label=None):
        if False:
            i = 10
            return i + 15
        'Returns a copy.\n\n        Make a shallow copy of d.  If you want a deep copy of d,\n        use copy.deepcopy on the whole object.\n\n        label: string label for the new Hist\n\n        returns: new _DictWrapper with the same type\n        '
        new = copy.copy(self)
        new.d = copy.copy(self.d)
        new.label = label if label is not None else self.label
        return new

    def Scale(self, factor):
        if False:
            print('Hello World!')
        'Multiplies the values by a factor.\n\n        factor: what to multiply by\n\n        Returns: new object\n        '
        new = self.Copy()
        new.d.clear()
        for (val, prob) in self.Items():
            new.Set(val * factor, prob)
        return new

    def Log(self, m=None):
        if False:
            return 10
        'Log transforms the probabilities.\n        \n        Removes values with probability 0.\n\n        Normalizes so that the largest logprob is 0.\n        '
        if self.log:
            raise ValueError('Pmf/Hist already under a log transform')
        self.log = True
        if m is None:
            m = self.MaxLike()
        for (x, p) in self.d.items():
            if p:
                self.Set(x, math.log(p / m))
            else:
                self.Remove(x)

    def Exp(self, m=None):
        if False:
            return 10
        'Exponentiates the probabilities.\n\n        m: how much to shift the ps before exponentiating\n\n        If m is None, normalizes so that the largest prob is 1.\n        '
        if not self.log:
            raise ValueError('Pmf/Hist not under a log transform')
        self.log = False
        if m is None:
            m = self.MaxLike()
        for (x, p) in self.d.items():
            self.Set(x, math.exp(p - m))

    def GetDict(self):
        if False:
            for i in range(10):
                print('nop')
        'Gets the dictionary.'
        return self.d

    def SetDict(self, d):
        if False:
            i = 10
            return i + 15
        'Sets the dictionary.'
        self.d = d

    def Values(self):
        if False:
            print('Hello World!')
        'Gets an unsorted sequence of values.\n\n        Note: one source of confusion is that the keys of this\n        dictionary are the values of the Hist/Pmf, and the\n        values of the dictionary are frequencies/probabilities.\n        '
        return self.d.keys()

    def Items(self):
        if False:
            return 10
        'Gets an unsorted sequence of (value, freq/prob) pairs.'
        return self.d.items()

    def Render(self, **options):
        if False:
            return 10
        'Generates a sequence of points suitable for plotting.\n\n        Note: options are ignored\n\n        Returns:\n            tuple of (sorted value sequence, freq/prob sequence)\n        '
        if min(self.d.keys()) is np.nan:
            logging.warning('Hist: contains NaN, may not render correctly.')
        return zip(*sorted(self.Items()))

    def MakeCdf(self, label=None):
        if False:
            print('Hello World!')
        'Makes a Cdf.'
        label = label if label is not None else self.label
        return Cdf(self, label=label)

    def Print(self):
        if False:
            return 10
        'Prints the values and freqs/probs in ascending order.'
        for (val, prob) in sorted(self.d.items()):
            print(val, prob)

    def Set(self, x, y=0):
        if False:
            print('Hello World!')
        'Sets the freq/prob associated with the value x.\n\n        Args:\n            x: number value\n            y: number freq or prob\n        '
        self.d[x] = y

    def Incr(self, x, term=1):
        if False:
            i = 10
            return i + 15
        'Increments the freq/prob associated with the value x.\n\n        Args:\n            x: number value\n            term: how much to increment by\n        '
        self.d[x] = self.d.get(x, 0) + term

    def Mult(self, x, factor):
        if False:
            for i in range(10):
                print('nop')
        'Scales the freq/prob associated with the value x.\n\n        Args:\n            x: number value\n            factor: how much to multiply by\n        '
        self.d[x] = self.d.get(x, 0) * factor

    def Remove(self, x):
        if False:
            print('Hello World!')
        'Removes a value.\n\n        Throws an exception if the value is not there.\n\n        Args:\n            x: value to remove\n        '
        del self.d[x]

    def Total(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the total of the frequencies/probabilities in the map.'
        total = sum(self.d.values())
        return total

    def MaxLike(self):
        if False:
            print('Hello World!')
        'Returns the largest frequency/probability in the map.'
        return max(self.d.values())

    def Largest(self, n=10):
        if False:
            while True:
                i = 10
        'Returns the largest n values, with frequency/probability.\n\n        n: number of items to return\n        '
        return sorted(self.d.items(), reverse=True)[:n]

    def Smallest(self, n=10):
        if False:
            return 10
        'Returns the smallest n values, with frequency/probability.\n\n        n: number of items to return\n        '
        return sorted(self.d.items(), reverse=False)[:n]

class Hist(_DictWrapper):
    """Represents a histogram, which is a map from values to frequencies.

    Values can be any hashable type; frequencies are integer counters.
    """

    def Freq(self, x):
        if False:
            i = 10
            return i + 15
        'Gets the frequency associated with the value x.\n\n        Args:\n            x: number value\n\n        Returns:\n            int frequency\n        '
        return self.d.get(x, 0)

    def Freqs(self, xs):
        if False:
            print('Hello World!')
        'Gets frequencies for a sequence of values.'
        return [self.Freq(x) for x in xs]

    def IsSubset(self, other):
        if False:
            return 10
        'Checks whether the values in this histogram are a subset of\n        the values in the given histogram.'
        for (val, freq) in self.Items():
            if freq > other.Freq(val):
                return False
        return True

    def Subtract(self, other):
        if False:
            i = 10
            return i + 15
        'Subtracts the values in the given histogram from this histogram.'
        for (val, freq) in other.Items():
            self.Incr(val, -freq)

class Pmf(_DictWrapper):
    """Represents a probability mass function.
    
    Values can be any hashable type; probabilities are floating-point.
    Pmfs are not necessarily normalized.
    """

    def Prob(self, x, default=0):
        if False:
            while True:
                i = 10
        'Gets the probability associated with the value x.\n\n        Args:\n            x: number value\n            default: value to return if the key is not there\n\n        Returns:\n            float probability\n        '
        return self.d.get(x, default)

    def Probs(self, xs):
        if False:
            return 10
        'Gets probabilities for a sequence of values.'
        return [self.Prob(x) for x in xs]

    def Percentile(self, percentage):
        if False:
            for i in range(10):
                print('nop')
        'Computes a percentile of a given Pmf.\n\n        Note: this is not super efficient.  If you are planning\n        to compute more than a few percentiles, compute the Cdf.\n\n        percentage: float 0-100\n\n        returns: value from the Pmf\n        '
        p = percentage / 100.0
        total = 0
        for (val, prob) in sorted(self.Items()):
            total += prob
            if total >= p:
                return val

    def ProbGreater(self, x):
        if False:
            print('Hello World!')
        'Probability that a sample from this Pmf exceeds x.\n\n        x: number\n\n        returns: float probability\n        '
        if isinstance(x, _DictWrapper):
            return PmfProbGreater(self, x)
        else:
            t = [prob for (val, prob) in self.d.items() if val > x]
            return sum(t)

    def ProbLess(self, x):
        if False:
            print('Hello World!')
        'Probability that a sample from this Pmf is less than x.\n\n        x: number\n\n        returns: float probability\n        '
        if isinstance(x, _DictWrapper):
            return PmfProbLess(self, x)
        else:
            t = [prob for (val, prob) in self.d.items() if val < x]
            return sum(t)

    def __lt__(self, obj):
        if False:
            for i in range(10):
                print('nop')
        'Less than.\n\n        obj: number or _DictWrapper\n\n        returns: float probability\n        '
        return self.ProbLess(obj)

    def __gt__(self, obj):
        if False:
            print('Hello World!')
        'Greater than.\n\n        obj: number or _DictWrapper\n\n        returns: float probability\n        '
        return self.ProbGreater(obj)

    def __ge__(self, obj):
        if False:
            i = 10
            return i + 15
        'Greater than or equal.\n\n        obj: number or _DictWrapper\n\n        returns: float probability\n        '
        return 1 - (self < obj)

    def __le__(self, obj):
        if False:
            while True:
                i = 10
        'Less than or equal.\n\n        obj: number or _DictWrapper\n\n        returns: float probability\n        '
        return 1 - (self > obj)

    def Normalize(self, fraction=1.0):
        if False:
            while True:
                i = 10
        'Normalizes this PMF so the sum of all probs is fraction.\n\n        Args:\n            fraction: what the total should be after normalization\n\n        Returns: the total probability before normalizing\n        '
        if self.log:
            raise ValueError('Normalize: Pmf is under a log transform')
        total = self.Total()
        if total == 0.0:
            raise ValueError('Normalize: total probability is zero.')
        factor = fraction / total
        for x in self.d:
            self.d[x] *= factor
        return total

    def Random(self):
        if False:
            return 10
        'Chooses a random element from this PMF.\n\n        Note: this is not very efficient.  If you plan to call\n        this more than a few times, consider converting to a CDF.\n\n        Returns:\n            float value from the Pmf\n        '
        target = random.random()
        total = 0.0
        for (x, p) in self.d.items():
            total += p
            if total >= target:
                return x
        raise ValueError('Random: Pmf might not be normalized.')

    def Mean(self):
        if False:
            while True:
                i = 10
        'Computes the mean of a PMF.\n\n        Returns:\n            float mean\n        '
        mean = 0.0
        for (x, p) in self.d.items():
            mean += p * x
        return mean

    def Var(self, mu=None):
        if False:
            while True:
                i = 10
        'Computes the variance of a PMF.\n\n        mu: the point around which the variance is computed;\n                if omitted, computes the mean\n\n        returns: float variance\n        '
        if mu is None:
            mu = self.Mean()
        var = 0.0
        for (x, p) in self.d.items():
            var += p * (x - mu) ** 2
        return var

    def Std(self, mu=None):
        if False:
            i = 10
            return i + 15
        'Computes the standard deviation of a PMF.\n\n        mu: the point around which the variance is computed;\n                if omitted, computes the mean\n\n        returns: float standard deviation\n        '
        var = self.Var(mu)
        return math.sqrt(var)

    def MaximumLikelihood(self):
        if False:
            print('Hello World!')
        'Returns the value with the highest probability.\n\n        Returns: float probability\n        '
        (_, val) = max(((prob, val) for (val, prob) in self.Items()))
        return val

    def CredibleInterval(self, percentage=90):
        if False:
            i = 10
            return i + 15
        'Computes the central credible interval.\n\n        If percentage=90, computes the 90% CI.\n\n        Args:\n            percentage: float between 0 and 100\n\n        Returns:\n            sequence of two floats, low and high\n        '
        cdf = self.MakeCdf()
        return cdf.CredibleInterval(percentage)

    def __add__(self, other):
        if False:
            return 10
        'Computes the Pmf of the sum of values drawn from self and other.\n\n        other: another Pmf or a scalar\n\n        returns: new Pmf\n        '
        try:
            return self.AddPmf(other)
        except AttributeError:
            return self.AddConstant(other)

    def AddPmf(self, other):
        if False:
            return 10
        'Computes the Pmf of the sum of values drawn from self and other.\n\n        other: another Pmf\n\n        returns: new Pmf\n        '
        pmf = Pmf()
        for (v1, p1) in self.Items():
            for (v2, p2) in other.Items():
                pmf.Incr(v1 + v2, p1 * p2)
        return pmf

    def AddConstant(self, other):
        if False:
            return 10
        'Computes the Pmf of the sum a constant and values from self.\n\n        other: a number\n\n        returns: new Pmf\n        '
        pmf = Pmf()
        for (v1, p1) in self.Items():
            pmf.Set(v1 + other, p1)
        return pmf

    def __sub__(self, other):
        if False:
            while True:
                i = 10
        'Computes the Pmf of the diff of values drawn from self and other.\n\n        other: another Pmf\n\n        returns: new Pmf\n        '
        try:
            return self.SubPmf(other)
        except AttributeError:
            return self.AddConstant(-other)

    def SubPmf(self, other):
        if False:
            while True:
                i = 10
        'Computes the Pmf of the diff of values drawn from self and other.\n\n        other: another Pmf\n\n        returns: new Pmf\n        '
        pmf = Pmf()
        for (v1, p1) in self.Items():
            for (v2, p2) in other.Items():
                pmf.Incr(v1 - v2, p1 * p2)
        return pmf

    def __mul__(self, other):
        if False:
            for i in range(10):
                print('nop')
        'Computes the Pmf of the product of values drawn from self and other.\n\n        other: another Pmf\n\n        returns: new Pmf\n        '
        try:
            return self.MulPmf(other)
        except AttributeError:
            return self.MulConstant(other)

    def MulPmf(self, other):
        if False:
            print('Hello World!')
        'Computes the Pmf of the diff of values drawn from self and other.\n\n        other: another Pmf\n\n        returns: new Pmf\n        '
        pmf = Pmf()
        for (v1, p1) in self.Items():
            for (v2, p2) in other.Items():
                pmf.Incr(v1 * v2, p1 * p2)
        return pmf

    def MulConstant(self, other):
        if False:
            i = 10
            return i + 15
        'Computes the Pmf of the product of a constant and values from self.\n\n        other: a number\n\n        returns: new Pmf\n        '
        pmf = Pmf()
        for (v1, p1) in self.Items():
            pmf.Set(v1 * other, p1)
        return pmf

    def __div__(self, other):
        if False:
            i = 10
            return i + 15
        'Computes the Pmf of the ratio of values drawn from self and other.\n\n        other: another Pmf\n\n        returns: new Pmf\n        '
        try:
            return self.DivPmf(other)
        except AttributeError:
            return self.MulConstant(1 / other)
    __truediv__ = __div__

    def DivPmf(self, other):
        if False:
            while True:
                i = 10
        'Computes the Pmf of the ratio of values drawn from self and other.\n\n        other: another Pmf\n\n        returns: new Pmf\n        '
        pmf = Pmf()
        for (v1, p1) in self.Items():
            for (v2, p2) in other.Items():
                pmf.Incr(v1 / v2, p1 * p2)
        return pmf

    def Max(self, k):
        if False:
            return 10
        'Computes the CDF of the maximum of k selections from this dist.\n\n        k: int\n\n        returns: new Cdf\n        '
        cdf = self.MakeCdf()
        return cdf.Max(k)

class Joint(Pmf):
    """Represents a joint distribution.

    The values are sequences (usually tuples)
    """

    def Marginal(self, i, label=None):
        if False:
            print('Hello World!')
        'Gets the marginal distribution of the indicated variable.\n\n        i: index of the variable we want\n\n        Returns: Pmf\n        '
        pmf = Pmf(label=label)
        for (vs, prob) in self.Items():
            pmf.Incr(vs[i], prob)
        return pmf

    def Conditional(self, i, j, val, label=None):
        if False:
            i = 10
            return i + 15
        'Gets the conditional distribution of the indicated variable.\n\n        Distribution of vs[i], conditioned on vs[j] = val.\n\n        i: index of the variable we want\n        j: which variable is conditioned on\n        val: the value the jth variable has to have\n\n        Returns: Pmf\n        '
        pmf = Pmf(label=label)
        for (vs, prob) in self.Items():
            if vs[j] != val:
                continue
            pmf.Incr(vs[i], prob)
        pmf.Normalize()
        return pmf

    def MaxLikeInterval(self, percentage=90):
        if False:
            while True:
                i = 10
        'Returns the maximum-likelihood credible interval.\n\n        If percentage=90, computes a 90% CI containing the values\n        with the highest likelihoods.\n\n        percentage: float between 0 and 100\n\n        Returns: list of values from the suite\n        '
        interval = []
        total = 0
        t = [(prob, val) for (val, prob) in self.Items()]
        t.sort(reverse=True)
        for (prob, val) in t:
            interval.append(val)
            total += prob
            if total >= percentage / 100.0:
                break
        return interval

def MakeJoint(pmf1, pmf2):
    if False:
        i = 10
        return i + 15
    'Joint distribution of values from pmf1 and pmf2.\n\n    Assumes that the PMFs represent independent random variables.\n\n    Args:\n        pmf1: Pmf object\n        pmf2: Pmf object\n\n    Returns:\n        Joint pmf of value pairs\n    '
    joint = Joint()
    for (v1, p1) in pmf1.Items():
        for (v2, p2) in pmf2.Items():
            joint.Set((v1, v2), p1 * p2)
    return joint

def MakeHistFromList(t, label=None):
    if False:
        print('Hello World!')
    'Makes a histogram from an unsorted sequence of values.\n\n    Args:\n        t: sequence of numbers\n        label: string label for this histogram\n\n    Returns:\n        Hist object\n    '
    return Hist(t, label=label)

def MakeHistFromDict(d, label=None):
    if False:
        print('Hello World!')
    'Makes a histogram from a map from values to frequencies.\n\n    Args:\n        d: dictionary that maps values to frequencies\n        label: string label for this histogram\n\n    Returns:\n        Hist object\n    '
    return Hist(d, label)

def MakePmfFromList(t, label=None):
    if False:
        for i in range(10):
            print('nop')
    'Makes a PMF from an unsorted sequence of values.\n\n    Args:\n        t: sequence of numbers\n        label: string label for this PMF\n\n    Returns:\n        Pmf object\n    '
    return Pmf(t, label=label)

def MakePmfFromDict(d, label=None):
    if False:
        while True:
            i = 10
    'Makes a PMF from a map from values to probabilities.\n\n    Args:\n        d: dictionary that maps values to probabilities\n        label: string label for this PMF\n\n    Returns:\n        Pmf object\n    '
    return Pmf(d, label=label)

def MakePmfFromItems(t, label=None):
    if False:
        return 10
    'Makes a PMF from a sequence of value-probability pairs\n\n    Args:\n        t: sequence of value-probability pairs\n        label: string label for this PMF\n\n    Returns:\n        Pmf object\n    '
    return Pmf(dict(t), label=label)

def MakePmfFromHist(hist, label=None):
    if False:
        while True:
            i = 10
    'Makes a normalized PMF from a Hist object.\n\n    Args:\n        hist: Hist object\n        label: string label\n\n    Returns:\n        Pmf object\n    '
    if label is None:
        label = hist.label
    return Pmf(hist, label=label)

def MakeMixture(metapmf, label='mix'):
    if False:
        i = 10
        return i + 15
    'Make a mixture distribution.\n\n    Args:\n      metapmf: Pmf that maps from Pmfs to probs.\n      label: string label for the new Pmf.\n\n    Returns: Pmf object.\n    '
    mix = Pmf(label=label)
    for (pmf, p1) in metapmf.Items():
        for (x, p2) in pmf.Items():
            mix.Incr(x, p1 * p2)
    return mix

def MakeUniformPmf(low, high, n):
    if False:
        print('Hello World!')
    'Make a uniform Pmf.\n\n    low: lowest value (inclusive)\n    high: highest value (inclusize)\n    n: number of values\n    '
    pmf = Pmf()
    for x in np.linspace(low, high, n):
        pmf.Set(x, 1)
    pmf.Normalize()
    return pmf

class Cdf(object):
    """Represents a cumulative distribution function.

    Attributes:
        xs: sequence of values
        ps: sequence of probabilities
        label: string used as a graph label.
    """

    def __init__(self, obj=None, ps=None, label=None):
        if False:
            for i in range(10):
                print('nop')
        'Initializes.\n        \n        If ps is provided, obj must be the corresponding list of values.\n\n        obj: Hist, Pmf, Cdf, Pdf, dict, pandas Series, list of pairs\n        ps: list of cumulative probabilities\n        label: string label\n        '
        self.label = label if label is not None else '_nolegend_'
        if isinstance(obj, (_DictWrapper, Cdf, Pdf)):
            if not label:
                self.label = label if label is not None else obj.label
        if obj is None:
            self.xs = np.asarray([])
            self.ps = np.asarray([])
            if ps is not None:
                logging.warning("Cdf: can't pass ps without also passing xs.")
            return
        elif ps is not None:
            if isinstance(ps, str):
                logging.warning("Cdf: ps can't be a string")
            self.xs = np.asarray(obj)
            self.ps = np.asarray(ps)
            return
        if isinstance(obj, Cdf):
            self.xs = copy.copy(obj.xs)
            self.ps = copy.copy(obj.ps)
            return
        if isinstance(obj, _DictWrapper):
            dw = obj
        else:
            dw = Hist(obj)
        if len(dw) == 0:
            self.xs = np.asarray([])
            self.ps = np.asarray([])
            return
        (xs, freqs) = zip(*sorted(dw.Items()))
        self.xs = np.asarray(xs)
        self.ps = np.cumsum(freqs, dtype=np.float)
        self.ps /= self.ps[-1]

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Cdf(%s, %s)' % (str(self.xs), str(self.ps))
    __repr__ = __str__

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.xs)

    def __getitem__(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self.Prob(x)

    def __setitem__(self):
        if False:
            print('Hello World!')
        raise UnimplementedMethodException()

    def __delitem__(self):
        if False:
            print('Hello World!')
        raise UnimplementedMethodException()

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return np.all(self.xs == other.xs) and np.all(self.ps == other.ps)

    def Copy(self, label=None):
        if False:
            for i in range(10):
                print('nop')
        'Returns a copy of this Cdf.\n\n        label: string label for the new Cdf\n        '
        if label is None:
            label = self.label
        return Cdf(list(self.xs), list(self.ps), label=label)

    def MakePmf(self, label=None):
        if False:
            while True:
                i = 10
        'Makes a Pmf.'
        if label is None:
            label = self.label
        return Pmf(self, label=label)

    def Values(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a sorted list of values.\n        '
        return self.xs

    def Items(self):
        if False:
            while True:
                i = 10
        'Returns a sorted sequence of (value, probability) pairs.\n\n        Note: in Python3, returns an iterator.\n        '
        a = self.ps
        b = np.roll(a, 1)
        b[0] = 0
        return zip(self.xs, a - b)

    def Shift(self, term):
        if False:
            print('Hello World!')
        'Adds a term to the xs.\n\n        term: how much to add\n        '
        new = self.Copy()
        new.xs = new.xs + term
        return new

    def Scale(self, factor):
        if False:
            print('Hello World!')
        'Multiplies the xs by a factor.\n\n        factor: what to multiply by\n        '
        new = self.Copy()
        new.xs = new.xs * factor
        return new

    def Prob(self, x):
        if False:
            return 10
        'Returns CDF(x), the probability that corresponds to value x.\n\n        Args:\n            x: number\n\n        Returns:\n            float probability\n        '
        if x < self.xs[0]:
            return 0.0
        index = bisect.bisect(self.xs, x)
        p = self.ps[index - 1]
        return p

    def Probs(self, xs):
        if False:
            i = 10
            return i + 15
        'Gets probabilities for a sequence of values.\n\n        xs: any sequence that can be converted to NumPy array\n\n        returns: NumPy array of cumulative probabilities\n        '
        xs = np.asarray(xs)
        index = np.searchsorted(self.xs, xs, side='right')
        ps = self.ps[index - 1]
        ps[xs < self.xs[0]] = 0.0
        return ps
    ProbArray = Probs

    def Value(self, p):
        if False:
            while True:
                i = 10
        'Returns InverseCDF(p), the value that corresponds to probability p.\n\n        Args:\n            p: number in the range [0, 1]\n\n        Returns:\n            number value\n        '
        if p < 0 or p > 1:
            raise ValueError('Probability p must be in range [0, 1]')
        index = bisect.bisect_left(self.ps, p)
        return self.xs[index]

    def ValueArray(self, ps):
        if False:
            print('Hello World!')
        'Returns InverseCDF(p), the value that corresponds to probability p.\n\n        Args:\n            ps: NumPy array of numbers in the range [0, 1]\n\n        Returns:\n            NumPy array of values\n        '
        ps = np.asarray(ps)
        if np.any(ps < 0) or np.any(ps > 1):
            raise ValueError('Probability p must be in range [0, 1]')
        index = np.searchsorted(self.ps, ps, side='left')
        return self.xs[index]

    def Percentile(self, p):
        if False:
            while True:
                i = 10
        'Returns the value that corresponds to percentile p.\n\n        Args:\n            p: number in the range [0, 100]\n\n        Returns:\n            number value\n        '
        return self.Value(p / 100.0)

    def PercentileRank(self, x):
        if False:
            while True:
                i = 10
        'Returns the percentile rank of the value x.\n\n        x: potential value in the CDF\n\n        returns: percentile rank in the range 0 to 100\n        '
        return self.Prob(x) * 100.0

    def Random(self):
        if False:
            i = 10
            return i + 15
        'Chooses a random value from this distribution.'
        return self.Value(random.random())

    def Sample(self, n):
        if False:
            while True:
                i = 10
        'Generates a random sample from this distribution.\n        \n        n: int length of the sample\n        returns: NumPy array\n        '
        ps = np.random.random(n)
        return self.ValueArray(ps)

    def Mean(self):
        if False:
            while True:
                i = 10
        'Computes the mean of a CDF.\n\n        Returns:\n            float mean\n        '
        old_p = 0
        total = 0.0
        for (x, new_p) in zip(self.xs, self.ps):
            p = new_p - old_p
            total += p * x
            old_p = new_p
        return total

    def CredibleInterval(self, percentage=90):
        if False:
            i = 10
            return i + 15
        'Computes the central credible interval.\n\n        If percentage=90, computes the 90% CI.\n\n        Args:\n            percentage: float between 0 and 100\n\n        Returns:\n            sequence of two floats, low and high\n        '
        prob = (1 - percentage / 100.0) / 2
        interval = (self.Value(prob), self.Value(1 - prob))
        return interval
    ConfidenceInterval = CredibleInterval

    def _Round(self, multiplier=1000.0):
        if False:
            i = 10
            return i + 15
        '\n        An entry is added to the cdf only if the percentile differs\n        from the previous value in a significant digit, where the number\n        of significant digits is determined by multiplier.  The\n        default is 1000, which keeps log10(1000) = 3 significant digits.\n        '
        raise UnimplementedMethodException()

    def Render(self, **options):
        if False:
            i = 10
            return i + 15
        'Generates a sequence of points suitable for plotting.\n\n        An empirical CDF is a step function; linear interpolation\n        can be misleading.\n\n        Note: options are ignored\n\n        Returns:\n            tuple of (xs, ps)\n        '

        def interleave(a, b):
            if False:
                return 10
            c = np.empty(a.shape[0] + b.shape[0])
            c[::2] = a
            c[1::2] = b
            return c
        a = np.array(self.xs)
        xs = interleave(a, a)
        shift_ps = np.roll(self.ps, 1)
        shift_ps[0] = 0
        ps = interleave(shift_ps, self.ps)
        return (xs, ps)

    def Max(self, k):
        if False:
            print('Hello World!')
        'Computes the CDF of the maximum of k selections from this dist.\n\n        k: int\n\n        returns: new Cdf\n        '
        cdf = self.Copy()
        cdf.ps **= k
        return cdf

def MakeCdfFromItems(items, label=None):
    if False:
        while True:
            i = 10
    'Makes a cdf from an unsorted sequence of (value, frequency) pairs.\n\n    Args:\n        items: unsorted sequence of (value, frequency) pairs\n        label: string label for this CDF\n\n    Returns:\n        cdf: list of (value, fraction) pairs\n    '
    return Cdf(dict(items), label=label)

def MakeCdfFromDict(d, label=None):
    if False:
        while True:
            i = 10
    'Makes a CDF from a dictionary that maps values to frequencies.\n\n    Args:\n       d: dictionary that maps values to frequencies.\n       label: string label for the data.\n\n    Returns:\n        Cdf object\n    '
    return Cdf(d, label=label)

def MakeCdfFromList(seq, label=None):
    if False:
        return 10
    'Creates a CDF from an unsorted sequence.\n\n    Args:\n        seq: unsorted sequence of sortable values\n        label: string label for the cdf\n\n    Returns:\n       Cdf object\n    '
    return Cdf(seq, label=label)

def MakeCdfFromHist(hist, label=None):
    if False:
        return 10
    'Makes a CDF from a Hist object.\n\n    Args:\n       hist: Pmf.Hist object\n       label: string label for the data.\n\n    Returns:\n        Cdf object\n    '
    if label is None:
        label = hist.label
    return Cdf(hist, label=label)

def MakeCdfFromPmf(pmf, label=None):
    if False:
        return 10
    'Makes a CDF from a Pmf object.\n\n    Args:\n       pmf: Pmf.Pmf object\n       label: string label for the data.\n\n    Returns:\n        Cdf object\n    '
    if label is None:
        label = pmf.label
    return Cdf(pmf, label=label)

class UnimplementedMethodException(Exception):
    """Exception if someone calls a method that should be overridden."""

class Suite(Pmf):
    """Represents a suite of hypotheses and their probabilities."""

    def Update(self, data):
        if False:
            print('Hello World!')
        'Updates each hypothesis based on the data.\n\n        data: any representation of the data\n\n        returns: the normalizing constant\n        '
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)
        return self.Normalize()

    def LogUpdate(self, data):
        if False:
            i = 10
            return i + 15
        'Updates a suite of hypotheses based on new data.\n\n        Modifies the suite directly; if you want to keep the original, make\n        a copy.\n\n        Note: unlike Update, LogUpdate does not normalize.\n\n        Args:\n            data: any representation of the data\n        '
        for hypo in self.Values():
            like = self.LogLikelihood(data, hypo)
            self.Incr(hypo, like)

    def UpdateSet(self, dataset):
        if False:
            for i in range(10):
                print('nop')
        'Updates each hypothesis based on the dataset.\n\n        This is more efficient than calling Update repeatedly because\n        it waits until the end to Normalize.\n\n        Modifies the suite directly; if you want to keep the original, make\n        a copy.\n\n        dataset: a sequence of data\n\n        returns: the normalizing constant\n        '
        for data in dataset:
            for hypo in self.Values():
                like = self.Likelihood(data, hypo)
                self.Mult(hypo, like)
        return self.Normalize()

    def LogUpdateSet(self, dataset):
        if False:
            return 10
        'Updates each hypothesis based on the dataset.\n\n        Modifies the suite directly; if you want to keep the original, make\n        a copy.\n\n        dataset: a sequence of data\n\n        returns: None\n        '
        for data in dataset:
            self.LogUpdate(data)

    def Likelihood(self, data, hypo):
        if False:
            print('Hello World!')
        'Computes the likelihood of the data under the hypothesis.\n\n        hypo: some representation of the hypothesis\n        data: some representation of the data\n        '
        raise UnimplementedMethodException()

    def LogLikelihood(self, data, hypo):
        if False:
            print('Hello World!')
        'Computes the log likelihood of the data under the hypothesis.\n\n        hypo: some representation of the hypothesis\n        data: some representation of the data\n        '
        raise UnimplementedMethodException()

    def Print(self):
        if False:
            while True:
                i = 10
        'Prints the hypotheses and their probabilities.'
        for (hypo, prob) in sorted(self.Items()):
            print(hypo, prob)

    def MakeOdds(self):
        if False:
            return 10
        'Transforms from probabilities to odds.\n\n        Values with prob=0 are removed.\n        '
        for (hypo, prob) in self.Items():
            if prob:
                self.Set(hypo, Odds(prob))
            else:
                self.Remove(hypo)

    def MakeProbs(self):
        if False:
            return 10
        'Transforms from odds to probabilities.'
        for (hypo, odds) in self.Items():
            self.Set(hypo, Probability(odds))

def MakeSuiteFromList(t, label=None):
    if False:
        print('Hello World!')
    'Makes a suite from an unsorted sequence of values.\n\n    Args:\n        t: sequence of numbers\n        label: string label for this suite\n\n    Returns:\n        Suite object\n    '
    hist = MakeHistFromList(t, label=label)
    d = hist.GetDict()
    return MakeSuiteFromDict(d)

def MakeSuiteFromHist(hist, label=None):
    if False:
        return 10
    'Makes a normalized suite from a Hist object.\n\n    Args:\n        hist: Hist object\n        label: string label\n\n    Returns:\n        Suite object\n    '
    if label is None:
        label = hist.label
    d = dict(hist.GetDict())
    return MakeSuiteFromDict(d, label)

def MakeSuiteFromDict(d, label=None):
    if False:
        while True:
            i = 10
    'Makes a suite from a map from values to probabilities.\n\n    Args:\n        d: dictionary that maps values to probabilities\n        label: string label for this suite\n\n    Returns:\n        Suite object\n    '
    suite = Suite(label=label)
    suite.SetDict(d)
    suite.Normalize()
    return suite

class Pdf(object):
    """Represents a probability density function (PDF)."""

    def Density(self, x):
        if False:
            return 10
        'Evaluates this Pdf at x.\n\n        Returns: float or NumPy array of probability density\n        '
        raise UnimplementedMethodException()

    def GetLinspace(self):
        if False:
            for i in range(10):
                print('nop')
        'Get a linspace for plotting.\n\n        Not all subclasses of Pdf implement this.\n\n        Returns: numpy array\n        '
        raise UnimplementedMethodException()

    def MakePmf(self, **options):
        if False:
            return 10
        'Makes a discrete version of this Pdf.\n\n        options can include\n        label: string\n        low: low end of range\n        high: high end of range\n        n: number of places to evaluate\n\n        Returns: new Pmf\n        '
        label = options.pop('label', '')
        (xs, ds) = self.Render(**options)
        return Pmf(dict(zip(xs, ds)), label=label)

    def Render(self, **options):
        if False:
            print('Hello World!')
        'Generates a sequence of points suitable for plotting.\n\n        If options includes low and high, it must also include n;\n        in that case the density is evaluated an n locations between\n        low and high, including both.\n\n        If options includes xs, the density is evaluate at those location.\n\n        Otherwise, self.GetLinspace is invoked to provide the locations.\n\n        Returns:\n            tuple of (xs, densities)\n        '
        (low, high) = (options.pop('low', None), options.pop('high', None))
        if low is not None and high is not None:
            n = options.pop('n', 101)
            xs = np.linspace(low, high, n)
        else:
            xs = options.pop('xs', None)
            if xs is None:
                xs = self.GetLinspace()
        ds = self.Density(xs)
        return (xs, ds)

    def Items(self):
        if False:
            while True:
                i = 10
        'Generates a sequence of (value, probability) pairs.\n        '
        return zip(*self.Render())

class NormalPdf(Pdf):
    """Represents the PDF of a Normal distribution."""

    def __init__(self, mu=0, sigma=1, label=None):
        if False:
            for i in range(10):
                print('nop')
        'Constructs a Normal Pdf with given mu and sigma.\n\n        mu: mean\n        sigma: standard deviation\n        label: string\n        '
        self.mu = mu
        self.sigma = sigma
        self.label = label if label is not None else '_nolegend_'

    def __str__(self):
        if False:
            print('Hello World!')
        return 'NormalPdf(%f, %f)' % (self.mu, self.sigma)

    def GetLinspace(self):
        if False:
            i = 10
            return i + 15
        'Get a linspace for plotting.\n\n        Returns: numpy array\n        '
        (low, high) = (self.mu - 3 * self.sigma, self.mu + 3 * self.sigma)
        return np.linspace(low, high, 101)

    def Density(self, xs):
        if False:
            for i in range(10):
                print('nop')
        'Evaluates this Pdf at xs.\n\n        xs: scalar or sequence of floats\n\n        returns: float or NumPy array of probability density\n        '
        return stats.norm.pdf(xs, self.mu, self.sigma)

class ExponentialPdf(Pdf):
    """Represents the PDF of an exponential distribution."""

    def __init__(self, lam=1, label=None):
        if False:
            print('Hello World!')
        'Constructs an exponential Pdf with given parameter.\n\n        lam: rate parameter\n        label: string\n        '
        self.lam = lam
        self.label = label if label is not None else '_nolegend_'

    def __str__(self):
        if False:
            print('Hello World!')
        return 'ExponentialPdf(%f)' % self.lam

    def GetLinspace(self):
        if False:
            return 10
        'Get a linspace for plotting.\n\n        Returns: numpy array\n        '
        (low, high) = (0, 5.0 / self.lam)
        return np.linspace(low, high, 101)

    def Density(self, xs):
        if False:
            while True:
                i = 10
        'Evaluates this Pdf at xs.\n\n        xs: scalar or sequence of floats\n\n        returns: float or NumPy array of probability density\n        '
        return stats.expon.pdf(xs, scale=1.0 / self.lam)

class EstimatedPdf(Pdf):
    """Represents a PDF estimated by KDE."""

    def __init__(self, sample, label=None):
        if False:
            while True:
                i = 10
        'Estimates the density function based on a sample.\n\n        sample: sequence of data\n        label: string\n        '
        self.label = label if label is not None else '_nolegend_'
        self.kde = stats.gaussian_kde(sample)
        low = min(sample)
        high = max(sample)
        self.linspace = np.linspace(low, high, 101)

    def __str__(self):
        if False:
            return 10
        return 'EstimatedPdf(label=%s)' % str(self.label)

    def GetLinspace(self):
        if False:
            return 10
        'Get a linspace for plotting.\n\n        Returns: numpy array\n        '
        return self.linspace

    def Density(self, xs):
        if False:
            for i in range(10):
                print('nop')
        'Evaluates this Pdf at xs.\n\n        returns: float or NumPy array of probability density\n        '
        return self.kde.evaluate(xs)

def CredibleInterval(pmf, percentage=90):
    if False:
        i = 10
        return i + 15
    'Computes a credible interval for a given distribution.\n\n    If percentage=90, computes the 90% CI.\n\n    Args:\n        pmf: Pmf object representing a posterior distribution\n        percentage: float between 0 and 100\n\n    Returns:\n        sequence of two floats, low and high\n    '
    cdf = pmf.MakeCdf()
    prob = (1 - percentage / 100.0) / 2
    interval = (cdf.Value(prob), cdf.Value(1 - prob))
    return interval

def PmfProbLess(pmf1, pmf2):
    if False:
        i = 10
        return i + 15
    'Probability that a value from pmf1 is less than a value from pmf2.\n\n    Args:\n        pmf1: Pmf object\n        pmf2: Pmf object\n\n    Returns:\n        float probability\n    '
    total = 0.0
    for (v1, p1) in pmf1.Items():
        for (v2, p2) in pmf2.Items():
            if v1 < v2:
                total += p1 * p2
    return total

def PmfProbGreater(pmf1, pmf2):
    if False:
        print('Hello World!')
    'Probability that a value from pmf1 is less than a value from pmf2.\n\n    Args:\n        pmf1: Pmf object\n        pmf2: Pmf object\n\n    Returns:\n        float probability\n    '
    total = 0.0
    for (v1, p1) in pmf1.Items():
        for (v2, p2) in pmf2.Items():
            if v1 > v2:
                total += p1 * p2
    return total

def PmfProbEqual(pmf1, pmf2):
    if False:
        i = 10
        return i + 15
    'Probability that a value from pmf1 equals a value from pmf2.\n\n    Args:\n        pmf1: Pmf object\n        pmf2: Pmf object\n\n    Returns:\n        float probability\n    '
    total = 0.0
    for (v1, p1) in pmf1.Items():
        for (v2, p2) in pmf2.Items():
            if v1 == v2:
                total += p1 * p2
    return total

def RandomSum(dists):
    if False:
        print('Hello World!')
    'Chooses a random value from each dist and returns the sum.\n\n    dists: sequence of Pmf or Cdf objects\n\n    returns: numerical sum\n    '
    total = sum((dist.Random() for dist in dists))
    return total

def SampleSum(dists, n):
    if False:
        i = 10
        return i + 15
    'Draws a sample of sums from a list of distributions.\n\n    dists: sequence of Pmf or Cdf objects\n    n: sample size\n\n    returns: new Pmf of sums\n    '
    pmf = Pmf((RandomSum(dists) for i in range(n)))
    return pmf

def EvalNormalPdf(x, mu, sigma):
    if False:
        while True:
            i = 10
    'Computes the unnormalized PDF of the normal distribution.\n\n    x: value\n    mu: mean\n    sigma: standard deviation\n    \n    returns: float probability density\n    '
    return stats.norm.pdf(x, mu, sigma)

def MakeNormalPmf(mu, sigma, num_sigmas, n=201):
    if False:
        return 10
    'Makes a PMF discrete approx to a Normal distribution.\n    \n    mu: float mean\n    sigma: float standard deviation\n    num_sigmas: how many sigmas to extend in each direction\n    n: number of values in the Pmf\n\n    returns: normalized Pmf\n    '
    pmf = Pmf()
    low = mu - num_sigmas * sigma
    high = mu + num_sigmas * sigma
    for x in np.linspace(low, high, n):
        p = EvalNormalPdf(x, mu, sigma)
        pmf.Set(x, p)
    pmf.Normalize()
    return pmf

def EvalBinomialPmf(k, n, p):
    if False:
        while True:
            i = 10
    'Evaluates the binomial PMF.\n\n    Returns the probabily of k successes in n trials with probability p.\n    '
    return stats.binom.pmf(k, n, p)

def EvalHypergeomPmf(k, N, K, n):
    if False:
        i = 10
        return i + 15
    'Evaluates the hypergeometric PMF.\n\n    Returns the probabily of k successes in n trials from a population\n    N with K successes in it.\n    '
    return stats.hypergeom.pmf(k, N, K, n)

def EvalPoissonPmf(k, lam):
    if False:
        return 10
    'Computes the Poisson PMF.\n\n    k: number of events\n    lam: parameter lambda in events per unit time\n\n    returns: float probability\n    '
    return lam ** k * math.exp(-lam) / special.gamma(k + 1)

def MakePoissonPmf(lam, high, step=1):
    if False:
        return 10
    'Makes a PMF discrete approx to a Poisson distribution.\n\n    lam: parameter lambda in events per unit time\n    high: upper bound of the Pmf\n\n    returns: normalized Pmf\n    '
    pmf = Pmf()
    for k in range(0, high + 1, step):
        p = EvalPoissonPmf(k, lam)
        pmf.Set(k, p)
    pmf.Normalize()
    return pmf

def EvalExponentialPdf(x, lam):
    if False:
        print('Hello World!')
    'Computes the exponential PDF.\n\n    x: value\n    lam: parameter lambda in events per unit time\n\n    returns: float probability density\n    '
    return lam * math.exp(-lam * x)

def EvalExponentialCdf(x, lam):
    if False:
        for i in range(10):
            print('nop')
    'Evaluates CDF of the exponential distribution with parameter lam.'
    return 1 - math.exp(-lam * x)

def MakeExponentialPmf(lam, high, n=200):
    if False:
        while True:
            i = 10
    'Makes a PMF discrete approx to an exponential distribution.\n\n    lam: parameter lambda in events per unit time\n    high: upper bound\n    n: number of values in the Pmf\n\n    returns: normalized Pmf\n    '
    pmf = Pmf()
    for x in np.linspace(0, high, n):
        p = EvalExponentialPdf(x, lam)
        pmf.Set(x, p)
    pmf.Normalize()
    return pmf

def StandardNormalCdf(x):
    if False:
        return 10
    'Evaluates the CDF of the standard Normal distribution.\n    \n    See http://en.wikipedia.org/wiki/Normal_distribution\n    #Cumulative_distribution_function\n\n    Args:\n        x: float\n                \n    Returns:\n        float\n    '
    return (math.erf(x / ROOT2) + 1) / 2

def EvalNormalCdf(x, mu=0, sigma=1):
    if False:
        for i in range(10):
            print('nop')
    'Evaluates the CDF of the normal distribution.\n    \n    Args:\n        x: float\n\n        mu: mean parameter\n        \n        sigma: standard deviation parameter\n                \n    Returns:\n        float\n    '
    return stats.norm.cdf(x, loc=mu, scale=sigma)

def EvalNormalCdfInverse(p, mu=0, sigma=1):
    if False:
        print('Hello World!')
    'Evaluates the inverse CDF of the normal distribution.\n\n    See http://en.wikipedia.org/wiki/Normal_distribution#Quantile_function  \n\n    Args:\n        p: float\n\n        mu: mean parameter\n        \n        sigma: standard deviation parameter\n                \n    Returns:\n        float\n    '
    return stats.norm.ppf(p, loc=mu, scale=sigma)

def EvalLognormalCdf(x, mu=0, sigma=1):
    if False:
        print('Hello World!')
    'Evaluates the CDF of the lognormal distribution.\n    \n    x: float or sequence\n    mu: mean parameter\n    sigma: standard deviation parameter\n                \n    Returns: float or sequence\n    '
    return stats.lognorm.cdf(x, loc=mu, scale=sigma)

def RenderExpoCdf(lam, low, high, n=101):
    if False:
        i = 10
        return i + 15
    'Generates sequences of xs and ps for an exponential CDF.\n\n    lam: parameter\n    low: float\n    high: float\n    n: number of points to render\n\n    returns: numpy arrays (xs, ps)\n    '
    xs = np.linspace(low, high, n)
    ps = 1 - np.exp(-lam * xs)
    return (xs, ps)

def RenderNormalCdf(mu, sigma, low, high, n=101):
    if False:
        for i in range(10):
            print('nop')
    'Generates sequences of xs and ps for a Normal CDF.\n\n    mu: parameter\n    sigma: parameter\n    low: float\n    high: float\n    n: number of points to render\n\n    returns: numpy arrays (xs, ps)\n    '
    xs = np.linspace(low, high, n)
    ps = stats.norm.cdf(xs, mu, sigma)
    return (xs, ps)

def RenderParetoCdf(xmin, alpha, low, high, n=50):
    if False:
        while True:
            i = 10
    'Generates sequences of xs and ps for a Pareto CDF.\n\n    xmin: parameter\n    alpha: parameter\n    low: float\n    high: float\n    n: number of points to render\n\n    returns: numpy arrays (xs, ps)\n    '
    if low < xmin:
        low = xmin
    xs = np.linspace(low, high, n)
    ps = 1 - (xs / xmin) ** (-alpha)
    return (xs, ps)

class Beta(object):
    """Represents a Beta distribution.

    See http://en.wikipedia.org/wiki/Beta_distribution
    """

    def __init__(self, alpha=1, beta=1, label=None):
        if False:
            return 10
        'Initializes a Beta distribution.'
        self.alpha = alpha
        self.beta = beta
        self.label = label if label is not None else '_nolegend_'

    def Update(self, data):
        if False:
            for i in range(10):
                print('nop')
        'Updates a Beta distribution.\n\n        data: pair of int (heads, tails)\n        '
        (heads, tails) = data
        self.alpha += heads
        self.beta += tails

    def Mean(self):
        if False:
            while True:
                i = 10
        'Computes the mean of this distribution.'
        return self.alpha / (self.alpha + self.beta)

    def Random(self):
        if False:
            print('Hello World!')
        'Generates a random variate from this distribution.'
        return random.betavariate(self.alpha, self.beta)

    def Sample(self, n):
        if False:
            print('Hello World!')
        'Generates a random sample from this distribution.\n\n        n: int sample size\n        '
        size = (n,)
        return np.random.beta(self.alpha, self.beta, size)

    def EvalPdf(self, x):
        if False:
            print('Hello World!')
        'Evaluates the PDF at x.'
        return x ** (self.alpha - 1) * (1 - x) ** (self.beta - 1)

    def MakePmf(self, steps=101, label=None):
        if False:
            i = 10
            return i + 15
        'Returns a Pmf of this distribution.\n\n        Note: Normally, we just evaluate the PDF at a sequence\n        of points and treat the probability density as a probability\n        mass.\n\n        But if alpha or beta is less than one, we have to be\n        more careful because the PDF goes to infinity at x=0\n        and x=1.  In that case we evaluate the CDF and compute\n        differences.\n        '
        if self.alpha < 1 or self.beta < 1:
            cdf = self.MakeCdf()
            pmf = cdf.MakePmf()
            return pmf
        xs = [i / (steps - 1.0) for i in range(steps)]
        probs = [self.EvalPdf(x) for x in xs]
        pmf = Pmf(dict(zip(xs, probs)), label=label)
        return pmf

    def MakeCdf(self, steps=101):
        if False:
            print('Hello World!')
        'Returns the CDF of this distribution.'
        xs = [i / (steps - 1.0) for i in range(steps)]
        ps = [special.betainc(self.alpha, self.beta, x) for x in xs]
        cdf = Cdf(xs, ps)
        return cdf

class Dirichlet(object):
    """Represents a Dirichlet distribution.

    See http://en.wikipedia.org/wiki/Dirichlet_distribution
    """

    def __init__(self, n, conc=1, label=None):
        if False:
            print('Hello World!')
        'Initializes a Dirichlet distribution.\n\n        n: number of dimensions\n        conc: concentration parameter (smaller yields more concentration)\n        label: string label\n        '
        if n < 2:
            raise ValueError('A Dirichlet distribution with n<2 makes no sense')
        self.n = n
        self.params = np.ones(n, dtype=np.float) * conc
        self.label = label if label is not None else '_nolegend_'

    def Update(self, data):
        if False:
            print('Hello World!')
        'Updates a Dirichlet distribution.\n\n        data: sequence of observations, in order corresponding to params\n        '
        m = len(data)
        self.params[:m] += data

    def Random(self):
        if False:
            for i in range(10):
                print('nop')
        'Generates a random variate from this distribution.\n\n        Returns: normalized vector of fractions\n        '
        p = np.random.gamma(self.params)
        return p / p.sum()

    def Likelihood(self, data):
        if False:
            print('Hello World!')
        'Computes the likelihood of the data.\n\n        Selects a random vector of probabilities from this distribution.\n\n        Returns: float probability\n        '
        m = len(data)
        if self.n < m:
            return 0
        x = data
        p = self.Random()
        q = p[:m] ** x
        return q.prod()

    def LogLikelihood(self, data):
        if False:
            return 10
        'Computes the log likelihood of the data.\n\n        Selects a random vector of probabilities from this distribution.\n\n        Returns: float log probability\n        '
        m = len(data)
        if self.n < m:
            return float('-inf')
        x = self.Random()
        y = np.log(x[:m]) * data
        return y.sum()

    def MarginalBeta(self, i):
        if False:
            i = 10
            return i + 15
        'Computes the marginal distribution of the ith element.\n\n        See http://en.wikipedia.org/wiki/Dirichlet_distribution\n        #Marginal_distributions\n\n        i: int\n\n        Returns: Beta object\n        '
        alpha0 = self.params.sum()
        alpha = self.params[i]
        return Beta(alpha, alpha0 - alpha)

    def PredictivePmf(self, xs, label=None):
        if False:
            return 10
        'Makes a predictive distribution.\n\n        xs: values to go into the Pmf\n\n        Returns: Pmf that maps from x to the mean prevalence of x\n        '
        alpha0 = self.params.sum()
        ps = self.params / alpha0
        return Pmf(zip(xs, ps), label=label)

def BinomialCoef(n, k):
    if False:
        return 10
    'Compute the binomial coefficient "n choose k".\n\n    n: number of trials\n    k: number of successes\n\n    Returns: float\n    '
    return scipy.misc.comb(n, k)

def LogBinomialCoef(n, k):
    if False:
        return 10
    'Computes the log of the binomial coefficient.\n\n    http://math.stackexchange.com/questions/64716/\n    approximating-the-logarithm-of-the-binomial-coefficient\n\n    n: number of trials\n    k: number of successes\n\n    Returns: float\n    '
    return n * math.log(n) - k * math.log(k) - (n - k) * math.log(n - k)

def NormalProbability(ys, jitter=0.0):
    if False:
        for i in range(10):
            print('nop')
    'Generates data for a normal probability plot.\n\n    ys: sequence of values\n    jitter: float magnitude of jitter added to the ys \n\n    returns: numpy arrays xs, ys\n    '
    n = len(ys)
    xs = np.random.normal(0, 1, n)
    xs.sort()
    if jitter:
        ys = Jitter(ys, jitter)
    else:
        ys = np.array(ys)
    ys.sort()
    return (xs, ys)

def Jitter(values, jitter=0.5):
    if False:
        return 10
    'Jitters the values by adding a uniform variate in (-jitter, jitter).\n\n    values: sequence\n    jitter: scalar magnitude of jitter\n    \n    returns: new numpy array\n    '
    n = len(values)
    return np.random.uniform(-jitter, +jitter, n) + values

def NormalProbabilityPlot(sample, fit_color='0.8', **options):
    if False:
        for i in range(10):
            print('nop')
    'Makes a normal probability plot with a fitted line.\n\n    sample: sequence of numbers\n    fit_color: color string for the fitted line\n    options: passed along to Plot\n    '
    (xs, ys) = NormalProbability(sample)
    (mean, var) = MeanVar(sample)
    std = math.sqrt(var)
    fit = FitLine(xs, mean, std)
    thinkplot.Plot(*fit, color=fit_color, label='model')
    (xs, ys) = NormalProbability(sample)
    thinkplot.Plot(xs, ys, **options)

def Mean(xs):
    if False:
        for i in range(10):
            print('nop')
    'Computes mean.\n\n    xs: sequence of values\n\n    returns: float mean\n    '
    return np.mean(xs)

def Var(xs, mu=None, ddof=0):
    if False:
        i = 10
        return i + 15
    'Computes variance.\n\n    xs: sequence of values\n    mu: option known mean\n    ddof: delta degrees of freedom\n\n    returns: float\n    '
    xs = np.asarray(xs)
    if mu is None:
        mu = xs.mean()
    ds = xs - mu
    return np.dot(ds, ds) / (len(xs) - ddof)

def Std(xs, mu=None, ddof=0):
    if False:
        print('Hello World!')
    'Computes standard deviation.\n\n    xs: sequence of values\n    mu: option known mean\n    ddof: delta degrees of freedom\n\n    returns: float\n    '
    var = Var(xs, mu, ddof)
    return math.sqrt(var)

def MeanVar(xs, ddof=0):
    if False:
        print('Hello World!')
    'Computes mean and variance.\n\n    Based on http://stackoverflow.com/questions/19391149/\n    numpy-mean-and-variance-from-single-function\n\n    xs: sequence of values\n    ddof: delta degrees of freedom\n    \n    returns: pair of float, mean and var\n    '
    xs = np.asarray(xs)
    mean = xs.mean()
    s2 = Var(xs, mean, ddof)
    return (mean, s2)

def Trim(t, p=0.01):
    if False:
        for i in range(10):
            print('nop')
    'Trims the largest and smallest elements of t.\n\n    Args:\n        t: sequence of numbers\n        p: fraction of values to trim off each end\n\n    Returns:\n        sequence of values\n    '
    n = int(p * len(t))
    t = sorted(t)[n:-n]
    return t

def TrimmedMean(t, p=0.01):
    if False:
        return 10
    'Computes the trimmed mean of a sequence of numbers.\n\n    Args:\n        t: sequence of numbers\n        p: fraction of values to trim off each end\n\n    Returns:\n        float\n    '
    t = Trim(t, p)
    return Mean(t)

def TrimmedMeanVar(t, p=0.01):
    if False:
        while True:
            i = 10
    'Computes the trimmed mean and variance of a sequence of numbers.\n\n    Side effect: sorts the list.\n\n    Args:\n        t: sequence of numbers\n        p: fraction of values to trim off each end\n\n    Returns:\n        float\n    '
    t = Trim(t, p)
    (mu, var) = MeanVar(t)
    return (mu, var)

def CohenEffectSize(group1, group2):
    if False:
        while True:
            i = 10
    "Compute Cohen's d.\n\n    group1: Series or NumPy array\n    group2: Series or NumPy array\n\n    returns: float\n    "
    diff = group1.mean() - group2.mean()
    (n1, n2) = (len(group1), len(group2))
    var1 = group1.var()
    var2 = group2.var()
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / math.sqrt(pooled_var)
    return d

def Cov(xs, ys, meanx=None, meany=None):
    if False:
        i = 10
        return i + 15
    'Computes Cov(X, Y).\n\n    Args:\n        xs: sequence of values\n        ys: sequence of values\n        meanx: optional float mean of xs\n        meany: optional float mean of ys\n\n    Returns:\n        Cov(X, Y)\n    '
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)
    cov = np.dot(xs - meanx, ys - meany) / len(xs)
    return cov

def Corr(xs, ys):
    if False:
        i = 10
        return i + 15
    'Computes Corr(X, Y).\n\n    Args:\n        xs: sequence of values\n        ys: sequence of values\n\n    Returns:\n        Corr(X, Y)\n    '
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    (meanx, varx) = MeanVar(xs)
    (meany, vary) = MeanVar(ys)
    corr = Cov(xs, ys, meanx, meany) / math.sqrt(varx * vary)
    return corr

def SerialCorr(series, lag=1):
    if False:
        for i in range(10):
            print('nop')
    'Computes the serial correlation of a series.\n\n    series: Series\n    lag: integer number of intervals to shift\n\n    returns: float correlation\n    '
    xs = series[lag:]
    ys = series.shift(lag)[lag:]
    corr = Corr(xs, ys)
    return corr

def SpearmanCorr(xs, ys):
    if False:
        i = 10
        return i + 15
    "Computes Spearman's rank correlation.\n\n    Args:\n        xs: sequence of values\n        ys: sequence of values\n\n    Returns:\n        float Spearman's correlation\n    "
    xranks = pandas.Series(xs).rank()
    yranks = pandas.Series(ys).rank()
    return Corr(xranks, yranks)

def MapToRanks(t):
    if False:
        i = 10
        return i + 15
    'Returns a list of ranks corresponding to the elements in t.\n\n    Args:\n        t: sequence of numbers\n    \n    Returns:\n        list of integer ranks, starting at 1\n    '
    pairs = enumerate(t)
    sorted_pairs = sorted(pairs, key=itemgetter(1))
    ranked = enumerate(sorted_pairs)
    resorted = sorted(ranked, key=lambda trip: trip[1][0])
    ranks = [trip[0] + 1 for trip in resorted]
    return ranks

def LeastSquares(xs, ys):
    if False:
        print('Hello World!')
    'Computes a linear least squares fit for ys as a function of xs.\n\n    Args:\n        xs: sequence of values\n        ys: sequence of values\n\n    Returns:\n        tuple of (intercept, slope)\n    '
    (meanx, varx) = MeanVar(xs)
    meany = Mean(ys)
    slope = Cov(xs, ys, meanx, meany) / varx
    inter = meany - slope * meanx
    return (inter, slope)

def FitLine(xs, inter, slope):
    if False:
        return 10
    'Fits a line to the given data.\n\n    xs: sequence of x\n\n    returns: tuple of numpy arrays (sorted xs, fit ys)\n    '
    fit_xs = np.sort(xs)
    fit_ys = inter + slope * fit_xs
    return (fit_xs, fit_ys)

def Residuals(xs, ys, inter, slope):
    if False:
        return 10
    'Computes residuals for a linear fit with parameters inter and slope.\n\n    Args:\n        xs: independent variable\n        ys: dependent variable\n        inter: float intercept\n        slope: float slope\n\n    Returns:\n        list of residuals\n    '
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    res = ys - (inter + slope * xs)
    return res

def CoefDetermination(ys, res):
    if False:
        while True:
            i = 10
    'Computes the coefficient of determination (R^2) for given residuals.\n\n    Args:\n        ys: dependent variable\n        res: residuals\n        \n    Returns:\n        float coefficient of determination\n    '
    return 1 - Var(res) / Var(ys)

def CorrelatedGenerator(rho):
    if False:
        return 10
    'Generates standard normal variates with serial correlation.\n\n    rho: target coefficient of correlation\n\n    Returns: iterable\n    '
    x = random.gauss(0, 1)
    yield x
    sigma = math.sqrt(1 - rho ** 2)
    while True:
        x = random.gauss(x * rho, sigma)
        yield x

def CorrelatedNormalGenerator(mu, sigma, rho):
    if False:
        return 10
    'Generates normal variates with serial correlation.\n\n    mu: mean of variate\n    sigma: standard deviation of variate\n    rho: target coefficient of correlation\n\n    Returns: iterable\n    '
    for x in CorrelatedGenerator(rho):
        yield (x * sigma + mu)

def RawMoment(xs, k):
    if False:
        for i in range(10):
            print('nop')
    'Computes the kth raw moment of xs.\n    '
    return sum((x ** k for x in xs)) / len(xs)

def CentralMoment(xs, k):
    if False:
        while True:
            i = 10
    'Computes the kth central moment of xs.\n    '
    mean = RawMoment(xs, 1)
    return sum(((x - mean) ** k for x in xs)) / len(xs)

def StandardizedMoment(xs, k):
    if False:
        return 10
    'Computes the kth standardized moment of xs.\n    '
    var = CentralMoment(xs, 2)
    std = math.sqrt(var)
    return CentralMoment(xs, k) / std ** k

def Skewness(xs):
    if False:
        while True:
            i = 10
    'Computes skewness.\n    '
    return StandardizedMoment(xs, 3)

def Median(xs):
    if False:
        for i in range(10):
            print('nop')
    'Computes the median (50th percentile) of a sequence.\n\n    xs: sequence or anything else that can initialize a Cdf\n\n    returns: float\n    '
    cdf = Cdf(xs)
    return cdf.Value(0.5)

def IQR(xs):
    if False:
        while True:
            i = 10
    'Computes the interquartile of a sequence.\n\n    xs: sequence or anything else that can initialize a Cdf\n\n    returns: pair of floats\n    '
    cdf = Cdf(xs)
    return (cdf.Value(0.25), cdf.Value(0.75))

def PearsonMedianSkewness(xs):
    if False:
        i = 10
        return i + 15
    'Computes the Pearson median skewness.\n    '
    median = Median(xs)
    mean = RawMoment(xs, 1)
    var = CentralMoment(xs, 2)
    std = math.sqrt(var)
    gp = 3 * (mean - median) / std
    return gp

class FixedWidthVariables(object):
    """Represents a set of variables in a fixed width file."""

    def __init__(self, variables, index_base=0):
        if False:
            print('Hello World!')
        'Initializes.\n\n        variables: DataFrame\n        index_base: are the indices 0 or 1 based?\n\n        Attributes:\n        colspecs: list of (start, end) index tuples\n        names: list of string variable names\n        '
        self.variables = variables
        self.colspecs = variables[['start', 'end']] - index_base
        self.colspecs = self.colspecs.astype(np.int).values.tolist()
        self.names = variables['name']

    def ReadFixedWidth(self, filename, **options):
        if False:
            while True:
                i = 10
        'Reads a fixed width ASCII file.\n\n        filename: string filename\n\n        returns: DataFrame\n        '
        df = pandas.read_fwf(filename, colspecs=self.colspecs, names=self.names, **options)
        return df

def ReadStataDct(dct_file, **options):
    if False:
        for i in range(10):
            print('nop')
    'Reads a Stata dictionary file.\n\n    dct_file: string filename\n    options: dict of options passed to open()\n\n    returns: FixedWidthVariables object\n    '
    type_map = dict(byte=int, int=int, long=int, float=float, double=float)
    var_info = []
    for line in open(dct_file, **options):
        match = re.search('_column\\(([^)]*)\\)', line)
        if match:
            start = int(match.group(1))
            t = line.split()
            (vtype, name, fstring) = t[1:4]
            name = name.lower()
            if vtype.startswith('str'):
                vtype = str
            else:
                vtype = type_map[vtype]
            long_desc = ' '.join(t[4:]).strip('"')
            var_info.append((start, vtype, name, fstring, long_desc))
    columns = ['start', 'type', 'name', 'fstring', 'desc']
    variables = pandas.DataFrame(var_info, columns=columns)
    variables['end'] = variables.start.shift(-1)
    variables.loc[len(variables) - 1, 'end'] = 0
    dct = FixedWidthVariables(variables, index_base=1)
    return dct

def Resample(xs, n=None):
    if False:
        return 10
    'Draw a sample from xs with the same length as xs.\n\n    xs: sequence\n    n: sample size (default: len(xs))\n\n    returns: NumPy array\n    '
    if n is None:
        n = len(xs)
    return np.random.choice(xs, n, replace=True)

def SampleRows(df, nrows, replace=False):
    if False:
        print('Hello World!')
    'Choose a sample of rows from a DataFrame.\n\n    df: DataFrame\n    nrows: number of rows\n    replace: whether to sample with replacement\n\n    returns: DataDf\n    '
    indices = np.random.choice(df.index, nrows, replace=replace)
    sample = df.loc[indices]
    return sample

def ResampleRows(df):
    if False:
        while True:
            i = 10
    'Resamples rows from a DataFrame.\n\n    df: DataFrame\n\n    returns: DataFrame\n    '
    return SampleRows(df, len(df), replace=True)

def ResampleRowsWeighted(df, column='finalwgt'):
    if False:
        return 10
    'Resamples a DataFrame using probabilities proportional to given column.\n\n    df: DataFrame\n    column: string column name to use as weights\n\n    returns: DataFrame\n    '
    weights = df[column]
    cdf = Cdf(dict(weights))
    indices = cdf.Sample(len(weights))
    sample = df.loc[indices]
    return sample

def PercentileRow(array, p):
    if False:
        i = 10
        return i + 15
    'Selects the row from a sorted array that maps to percentile p.\n\n    p: float 0--100\n\n    returns: NumPy array (one row)\n    '
    (rows, cols) = array.shape
    index = int(rows * p / 100)
    return array[index,]

def PercentileRows(ys_seq, percents):
    if False:
        return 10
    'Given a collection of lines, selects percentiles along vertical axis.\n\n    For example, if ys_seq contains simulation results like ys as a\n    function of time, and percents contains (5, 95), the result would\n    be a 90% CI for each vertical slice of the simulation results.\n\n    ys_seq: sequence of lines (y values)\n    percents: list of percentiles (0-100) to select\n\n    returns: list of NumPy arrays, one for each percentile\n    '
    nrows = len(ys_seq)
    ncols = len(ys_seq[0])
    array = np.zeros((nrows, ncols))
    for (i, ys) in enumerate(ys_seq):
        array[i,] = ys
    array = np.sort(array, axis=0)
    rows = [PercentileRow(array, p) for p in percents]
    return rows

def Smooth(xs, sigma=2, **options):
    if False:
        while True:
            i = 10
    'Smooths a NumPy array with a Gaussian filter.\n\n    xs: sequence\n    sigma: standard deviation of the filter\n    '
    return ndimage.filters.gaussian_filter1d(xs, sigma, **options)

class HypothesisTest(object):
    """Represents a hypothesis test."""

    def __init__(self, data):
        if False:
            while True:
                i = 10
        'Initializes.\n\n        data: data in whatever form is relevant\n        '
        self.data = data
        self.MakeModel()
        self.actual = self.TestStatistic(data)
        self.test_stats = None
        self.test_cdf = None

    def PValue(self, iters=1000):
        if False:
            print('Hello World!')
        'Computes the distribution of the test statistic and p-value.\n\n        iters: number of iterations\n\n        returns: float p-value\n        '
        self.test_stats = [self.TestStatistic(self.RunModel()) for _ in range(iters)]
        self.test_cdf = Cdf(self.test_stats)
        count = sum((1 for x in self.test_stats if x >= self.actual))
        return count / iters

    def MaxTestStat(self):
        if False:
            print('Hello World!')
        'Returns the largest test statistic seen during simulations.\n        '
        return max(self.test_stats)

    def PlotCdf(self, label=None):
        if False:
            for i in range(10):
                print('nop')
        'Draws a Cdf with vertical lines at the observed test stat.\n        '

        def VertLine(x):
            if False:
                print('Hello World!')
            'Draws a vertical line at x.'
            thinkplot.Plot([x, x], [0, 1], color='0.8')
        VertLine(self.actual)
        thinkplot.Cdf(self.test_cdf, label=label)

    def TestStatistic(self, data):
        if False:
            return 10
        'Computes the test statistic.\n\n        data: data in whatever form is relevant        \n        '
        raise UnimplementedMethodException()

    def MakeModel(self):
        if False:
            return 10
        'Build a model of the null hypothesis.\n        '
        pass

    def RunModel(self):
        if False:
            while True:
                i = 10
        'Run the model of the null hypothesis.\n\n        returns: simulated data\n        '
        raise UnimplementedMethodException()

def main():
    if False:
        for i in range(10):
            print('nop')
    pass
if __name__ == '__main__':
    main()