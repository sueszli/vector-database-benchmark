"""
Provides scoring functions for a number of association measures through a
generic, abstract implementation in ``NgramAssocMeasures``, and n-specific
``BigramAssocMeasures`` and ``TrigramAssocMeasures``.
"""
import math as _math
from abc import ABCMeta, abstractmethod
from functools import reduce
_log2 = lambda x: _math.log2(x)
_ln = _math.log
_product = lambda s: reduce(lambda x, y: x * y, s)
_SMALL = 1e-20
try:
    from scipy.stats import fisher_exact
except ImportError:

    def fisher_exact(*_args, **_kwargs):
        if False:
            while True:
                i = 10
        raise NotImplementedError
NGRAM = 0
'Marginals index for the ngram count'
UNIGRAMS = -2
'Marginals index for a tuple of each unigram count'
TOTAL = -1
'Marginals index for the number of words in the data'

class NgramAssocMeasures(metaclass=ABCMeta):
    """
    An abstract class defining a collection of generic association measures.
    Each public method returns a score, taking the following arguments::

        score_fn(count_of_ngram,
                 (count_of_n-1gram_1, ..., count_of_n-1gram_j),
                 (count_of_n-2gram_1, ..., count_of_n-2gram_k),
                 ...,
                 (count_of_1gram_1, ..., count_of_1gram_n),
                 count_of_total_words)

    See ``BigramAssocMeasures`` and ``TrigramAssocMeasures``

    Inheriting classes should define a property _n, and a method _contingency
    which calculates contingency values from marginals in order for all
    association measures defined here to be usable.
    """
    _n = 0

    @staticmethod
    @abstractmethod
    def _contingency(*marginals):
        if False:
            i = 10
            return i + 15
        'Calculates values of a contingency table from marginal values.'
        raise NotImplementedError('The contingency table is not availablein the general ngram case')

    @staticmethod
    @abstractmethod
    def _marginals(*contingency):
        if False:
            for i in range(10):
                print('nop')
        'Calculates values of contingency table marginals from its values.'
        raise NotImplementedError('The contingency table is not availablein the general ngram case')

    @classmethod
    def _expected_values(cls, cont):
        if False:
            return 10
        'Calculates expected values for a contingency table.'
        n_all = sum(cont)
        bits = [1 << i for i in range(cls._n)]
        for i in range(len(cont)):
            yield (_product((sum((cont[x] for x in range(2 ** cls._n) if x & j == i & j)) for j in bits)) / n_all ** (cls._n - 1))

    @staticmethod
    def raw_freq(*marginals):
        if False:
            return 10
        'Scores ngrams by their frequency'
        return marginals[NGRAM] / marginals[TOTAL]

    @classmethod
    def student_t(cls, *marginals):
        if False:
            for i in range(10):
                print('nop')
        "Scores ngrams using Student's t test with independence hypothesis\n        for unigrams, as in Manning and Schutze 5.3.1.\n        "
        return (marginals[NGRAM] - _product(marginals[UNIGRAMS]) / marginals[TOTAL] ** (cls._n - 1)) / (marginals[NGRAM] + _SMALL) ** 0.5

    @classmethod
    def chi_sq(cls, *marginals):
        if False:
            for i in range(10):
                print('nop')
        "Scores ngrams using Pearson's chi-square as in Manning and Schutze\n        5.3.3.\n        "
        cont = cls._contingency(*marginals)
        exps = cls._expected_values(cont)
        return sum(((obs - exp) ** 2 / (exp + _SMALL) for (obs, exp) in zip(cont, exps)))

    @staticmethod
    def mi_like(*marginals, **kwargs):
        if False:
            i = 10
            return i + 15
        'Scores ngrams using a variant of mutual information. The keyword\n        argument power sets an exponent (default 3) for the numerator. No\n        logarithm of the result is calculated.\n        '
        return marginals[NGRAM] ** kwargs.get('power', 3) / _product(marginals[UNIGRAMS])

    @classmethod
    def pmi(cls, *marginals):
        if False:
            return 10
        'Scores ngrams by pointwise mutual information, as in Manning and\n        Schutze 5.4.\n        '
        return _log2(marginals[NGRAM] * marginals[TOTAL] ** (cls._n - 1)) - _log2(_product(marginals[UNIGRAMS]))

    @classmethod
    def likelihood_ratio(cls, *marginals):
        if False:
            while True:
                i = 10
        'Scores ngrams using likelihood ratios as in Manning and Schutze 5.3.4.'
        cont = cls._contingency(*marginals)
        return 2 * sum((obs * _ln(obs / (exp + _SMALL) + _SMALL) for (obs, exp) in zip(cont, cls._expected_values(cont))))

    @classmethod
    def poisson_stirling(cls, *marginals):
        if False:
            return 10
        'Scores ngrams using the Poisson-Stirling measure.'
        exp = _product(marginals[UNIGRAMS]) / marginals[TOTAL] ** (cls._n - 1)
        return marginals[NGRAM] * (_log2(marginals[NGRAM] / exp) - 1)

    @classmethod
    def jaccard(cls, *marginals):
        if False:
            print('Hello World!')
        'Scores ngrams using the Jaccard index.'
        cont = cls._contingency(*marginals)
        return cont[0] / sum(cont[:-1])

class BigramAssocMeasures(NgramAssocMeasures):
    """
    A collection of bigram association measures. Each association measure
    is provided as a function with three arguments::

        bigram_score_fn(n_ii, (n_ix, n_xi), n_xx)

    The arguments constitute the marginals of a contingency table, counting
    the occurrences of particular events in a corpus. The letter i in the
    suffix refers to the appearance of the word in question, while x indicates
    the appearance of any word. Thus, for example:

    - n_ii counts ``(w1, w2)``, i.e. the bigram being scored
    - n_ix counts ``(w1, *)``
    - n_xi counts ``(*, w2)``
    - n_xx counts ``(*, *)``, i.e. any bigram

    This may be shown with respect to a contingency table::

                w1    ~w1
             ------ ------
         w2 | n_ii | n_oi | = n_xi
             ------ ------
        ~w2 | n_io | n_oo |
             ------ ------
             = n_ix        TOTAL = n_xx
    """
    _n = 2

    @staticmethod
    def _contingency(n_ii, n_ix_xi_tuple, n_xx):
        if False:
            return 10
        'Calculates values of a bigram contingency table from marginal values.'
        (n_ix, n_xi) = n_ix_xi_tuple
        n_oi = n_xi - n_ii
        n_io = n_ix - n_ii
        return (n_ii, n_oi, n_io, n_xx - n_ii - n_oi - n_io)

    @staticmethod
    def _marginals(n_ii, n_oi, n_io, n_oo):
        if False:
            print('Hello World!')
        'Calculates values of contingency table marginals from its values.'
        return (n_ii, (n_oi + n_ii, n_io + n_ii), n_oo + n_oi + n_io + n_ii)

    @staticmethod
    def _expected_values(cont):
        if False:
            return 10
        'Calculates expected values for a contingency table.'
        n_xx = sum(cont)
        for i in range(4):
            yield ((cont[i] + cont[i ^ 1]) * (cont[i] + cont[i ^ 2]) / n_xx)

    @classmethod
    def phi_sq(cls, *marginals):
        if False:
            while True:
                i = 10
        'Scores bigrams using phi-square, the square of the Pearson correlation\n        coefficient.\n        '
        (n_ii, n_io, n_oi, n_oo) = cls._contingency(*marginals)
        return (n_ii * n_oo - n_io * n_oi) ** 2 / ((n_ii + n_io) * (n_ii + n_oi) * (n_io + n_oo) * (n_oi + n_oo))

    @classmethod
    def chi_sq(cls, n_ii, n_ix_xi_tuple, n_xx):
        if False:
            print('Hello World!')
        'Scores bigrams using chi-square, i.e. phi-sq multiplied by the number\n        of bigrams, as in Manning and Schutze 5.3.3.\n        '
        (n_ix, n_xi) = n_ix_xi_tuple
        return n_xx * cls.phi_sq(n_ii, (n_ix, n_xi), n_xx)

    @classmethod
    def fisher(cls, *marginals):
        if False:
            print('Hello World!')
        "Scores bigrams using Fisher's Exact Test (Pedersen 1996).  Less\n        sensitive to small counts than PMI or Chi Sq, but also more expensive\n        to compute. Requires scipy.\n        "
        (n_ii, n_io, n_oi, n_oo) = cls._contingency(*marginals)
        (odds, pvalue) = fisher_exact([[n_ii, n_io], [n_oi, n_oo]], alternative='less')
        return pvalue

    @staticmethod
    def dice(n_ii, n_ix_xi_tuple, n_xx):
        if False:
            return 10
        "Scores bigrams using Dice's coefficient."
        (n_ix, n_xi) = n_ix_xi_tuple
        return 2 * n_ii / (n_ix + n_xi)

class TrigramAssocMeasures(NgramAssocMeasures):
    """
    A collection of trigram association measures. Each association measure
    is provided as a function with four arguments::

        trigram_score_fn(n_iii,
                         (n_iix, n_ixi, n_xii),
                         (n_ixx, n_xix, n_xxi),
                         n_xxx)

    The arguments constitute the marginals of a contingency table, counting
    the occurrences of particular events in a corpus. The letter i in the
    suffix refers to the appearance of the word in question, while x indicates
    the appearance of any word. Thus, for example:

    - n_iii counts ``(w1, w2, w3)``, i.e. the trigram being scored
    - n_ixx counts ``(w1, *, *)``
    - n_xxx counts ``(*, *, *)``, i.e. any trigram
    """
    _n = 3

    @staticmethod
    def _contingency(n_iii, n_iix_tuple, n_ixx_tuple, n_xxx):
        if False:
            while True:
                i = 10
        'Calculates values of a trigram contingency table (or cube) from\n        marginal values.\n        >>> TrigramAssocMeasures._contingency(1, (1, 1, 1), (1, 73, 1), 2000)\n        (1, 0, 0, 0, 0, 72, 0, 1927)\n        '
        (n_iix, n_ixi, n_xii) = n_iix_tuple
        (n_ixx, n_xix, n_xxi) = n_ixx_tuple
        n_oii = n_xii - n_iii
        n_ioi = n_ixi - n_iii
        n_iio = n_iix - n_iii
        n_ooi = n_xxi - n_iii - n_oii - n_ioi
        n_oio = n_xix - n_iii - n_oii - n_iio
        n_ioo = n_ixx - n_iii - n_ioi - n_iio
        n_ooo = n_xxx - n_iii - n_oii - n_ioi - n_iio - n_ooi - n_oio - n_ioo
        return (n_iii, n_oii, n_ioi, n_ooi, n_iio, n_oio, n_ioo, n_ooo)

    @staticmethod
    def _marginals(*contingency):
        if False:
            return 10
        'Calculates values of contingency table marginals from its values.\n        >>> TrigramAssocMeasures._marginals(1, 0, 0, 0, 0, 72, 0, 1927)\n        (1, (1, 1, 1), (1, 73, 1), 2000)\n        '
        (n_iii, n_oii, n_ioi, n_ooi, n_iio, n_oio, n_ioo, n_ooo) = contingency
        return (n_iii, (n_iii + n_iio, n_iii + n_ioi, n_iii + n_oii), (n_iii + n_ioi + n_iio + n_ioo, n_iii + n_oii + n_iio + n_oio, n_iii + n_oii + n_ioi + n_ooi), sum(contingency))

class QuadgramAssocMeasures(NgramAssocMeasures):
    """
    A collection of quadgram association measures. Each association measure
    is provided as a function with five arguments::

        trigram_score_fn(n_iiii,
                        (n_iiix, n_iixi, n_ixii, n_xiii),
                        (n_iixx, n_ixix, n_ixxi, n_xixi, n_xxii, n_xiix),
                        (n_ixxx, n_xixx, n_xxix, n_xxxi),
                        n_all)

    The arguments constitute the marginals of a contingency table, counting
    the occurrences of particular events in a corpus. The letter i in the
    suffix refers to the appearance of the word in question, while x indicates
    the appearance of any word. Thus, for example:

    - n_iiii counts ``(w1, w2, w3, w4)``, i.e. the quadgram being scored
    - n_ixxi counts ``(w1, *, *, w4)``
    - n_xxxx counts ``(*, *, *, *)``, i.e. any quadgram
    """
    _n = 4

    @staticmethod
    def _contingency(n_iiii, n_iiix_tuple, n_iixx_tuple, n_ixxx_tuple, n_xxxx):
        if False:
            i = 10
            return i + 15
        'Calculates values of a quadgram contingency table from\n        marginal values.\n        '
        (n_iiix, n_iixi, n_ixii, n_xiii) = n_iiix_tuple
        (n_iixx, n_ixix, n_ixxi, n_xixi, n_xxii, n_xiix) = n_iixx_tuple
        (n_ixxx, n_xixx, n_xxix, n_xxxi) = n_ixxx_tuple
        n_oiii = n_xiii - n_iiii
        n_ioii = n_ixii - n_iiii
        n_iioi = n_iixi - n_iiii
        n_ooii = n_xxii - n_iiii - n_oiii - n_ioii
        n_oioi = n_xixi - n_iiii - n_oiii - n_iioi
        n_iooi = n_ixxi - n_iiii - n_ioii - n_iioi
        n_oooi = n_xxxi - n_iiii - n_oiii - n_ioii - n_iioi - n_ooii - n_iooi - n_oioi
        n_iiio = n_iiix - n_iiii
        n_oiio = n_xiix - n_iiii - n_oiii - n_iiio
        n_ioio = n_ixix - n_iiii - n_ioii - n_iiio
        n_ooio = n_xxix - n_iiii - n_oiii - n_ioii - n_iiio - n_ooii - n_ioio - n_oiio
        n_iioo = n_iixx - n_iiii - n_iioi - n_iiio
        n_oioo = n_xixx - n_iiii - n_oiii - n_iioi - n_iiio - n_oioi - n_oiio - n_iioo
        n_iooo = n_ixxx - n_iiii - n_ioii - n_iioi - n_iiio - n_iooi - n_iioo - n_ioio
        n_oooo = n_xxxx - n_iiii - n_oiii - n_ioii - n_iioi - n_ooii - n_oioi - n_iooi - n_oooi - n_iiio - n_oiio - n_ioio - n_ooio - n_iioo - n_oioo - n_iooo
        return (n_iiii, n_oiii, n_ioii, n_ooii, n_iioi, n_oioi, n_iooi, n_oooi, n_iiio, n_oiio, n_ioio, n_ooio, n_iioo, n_oioo, n_iooo, n_oooo)

    @staticmethod
    def _marginals(*contingency):
        if False:
            print('Hello World!')
        'Calculates values of contingency table marginals from its values.\n        QuadgramAssocMeasures._marginals(1, 0, 2, 46, 552, 825, 2577, 34967, 1, 0, 2, 48, 7250, 9031, 28585, 356653)\n        (1, (2, 553, 3, 1), (7804, 6, 3132, 1378, 49, 2), (38970, 17660, 100, 38970), 440540)\n        '
        (n_iiii, n_oiii, n_ioii, n_ooii, n_iioi, n_oioi, n_iooi, n_oooi, n_iiio, n_oiio, n_ioio, n_ooio, n_iioo, n_oioo, n_iooo, n_oooo) = contingency
        n_iiix = n_iiii + n_iiio
        n_iixi = n_iiii + n_iioi
        n_ixii = n_iiii + n_ioii
        n_xiii = n_iiii + n_oiii
        n_iixx = n_iiii + n_iioi + n_iiio + n_iioo
        n_ixix = n_iiii + n_ioii + n_iiio + n_ioio
        n_ixxi = n_iiii + n_ioii + n_iioi + n_iooi
        n_xixi = n_iiii + n_oiii + n_iioi + n_oioi
        n_xxii = n_iiii + n_oiii + n_ioii + n_ooii
        n_xiix = n_iiii + n_oiii + n_iiio + n_oiio
        n_ixxx = n_iiii + n_ioii + n_iioi + n_iiio + n_iooi + n_iioo + n_ioio + n_iooo
        n_xixx = n_iiii + n_oiii + n_iioi + n_iiio + n_oioi + n_oiio + n_iioo + n_oioo
        n_xxix = n_iiii + n_oiii + n_ioii + n_iiio + n_ooii + n_ioio + n_oiio + n_ooio
        n_xxxi = n_iiii + n_oiii + n_ioii + n_iioi + n_ooii + n_iooi + n_oioi + n_oooi
        n_all = sum(contingency)
        return (n_iiii, (n_iiix, n_iixi, n_ixii, n_xiii), (n_iixx, n_ixix, n_ixxi, n_xixi, n_xxii, n_xiix), (n_ixxx, n_xixx, n_xxix, n_xxxi), n_all)

class ContingencyMeasures:
    """Wraps NgramAssocMeasures classes such that the arguments of association
    measures are contingency table values rather than marginals.
    """

    def __init__(self, measures):
        if False:
            print('Hello World!')
        'Constructs a ContingencyMeasures given a NgramAssocMeasures class'
        self.__class__.__name__ = 'Contingency' + measures.__class__.__name__
        for k in dir(measures):
            if k.startswith('__'):
                continue
            v = getattr(measures, k)
            if not k.startswith('_'):
                v = self._make_contingency_fn(measures, v)
            setattr(self, k, v)

    @staticmethod
    def _make_contingency_fn(measures, old_fn):
        if False:
            print('Hello World!')
        'From an association measure function, produces a new function which\n        accepts contingency table values as its arguments.\n        '

        def res(*contingency):
            if False:
                for i in range(10):
                    print('nop')
            return old_fn(*measures._marginals(*contingency))
        res.__doc__ = old_fn.__doc__
        res.__name__ = old_fn.__name__
        return res