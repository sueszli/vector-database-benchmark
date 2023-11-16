from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

def f_score(evaluated_sentences, reference_sentences, weight=1.0):
    if False:
        while True:
            i = 10
    '\n    Computation of F-Score measure. It is computed as\n    F(E) = ( (W^2 + 1) * P(E) * R(E) ) / ( W^2 * P(E) + R(E) ), where:\n\n    - P(E) is precision metrics of extract E.\n    - R(E) is recall metrics of extract E.\n    - W is a weighting factor that favours P(E) metrics\n      when W > 1 and favours R(E) metrics when W < 1.\n\n    If W = 1.0 (default value) basic F-Score is computed.\n    It is equivalent to F(E) = (2 * P(E) * R(E)) / (P(E) + R(E)).\n\n    :parameter iterable evaluated_sentences:\n        Sentences of evaluated extract.\n    :parameter iterable reference_sentences:\n        Sentences of reference extract.\n    :returns float:\n        Returns 0.0 <= P(E) <= 1.0\n    '
    p = precision(evaluated_sentences, reference_sentences)
    r = recall(evaluated_sentences, reference_sentences)
    weight **= 2
    denominator = weight * p + r
    if denominator == 0.0:
        return 0.0
    else:
        return (weight + 1) * p * r / denominator

def precision(evaluated_sentences, reference_sentences):
    if False:
        print('Hello World!')
    '\n    Intrinsic method of evaluation for extracts. It is computed as\n    P(E) = A / B, where:\n\n    - A is count of common sentences occurring in both extracts.\n    - B is count of sentences in evaluated extract.\n\n    :parameter iterable evaluated_sentences:\n        Sentences of evaluated extract.\n    :parameter iterable reference_sentences:\n        Sentences of reference extract.\n    :returns float:\n        Returns 0.0 <= P(E) <= 1.0\n    '
    return _divide_evaluation(reference_sentences, evaluated_sentences)

def recall(evaluated_sentences, reference_sentences):
    if False:
        return 10
    '\n    Intrinsic method of evaluation for extracts. It is computed as\n    R(E) = A / C, where:\n\n    - A is count of common sentences in both extracts.\n    - C is count of sentences in reference extract.\n\n    :parameter iterable evaluated_sentences:\n        Sentences of evaluated extract.\n    :parameter iterable reference_sentences:\n        Sentences of reference extract.\n    :returns float:\n        Returns 0.0 <= R(E) <= 1.0\n    '
    return _divide_evaluation(evaluated_sentences, reference_sentences)

def _divide_evaluation(numerator_sentences, denominator_sentences):
    if False:
        i = 10
        return i + 15
    denominator_sentences = frozenset(denominator_sentences)
    numerator_sentences = frozenset(numerator_sentences)
    if len(numerator_sentences) == 0 or len(denominator_sentences) == 0:
        raise ValueError('Both collections have to contain at least 1 sentence.')
    common_count = len(denominator_sentences & numerator_sentences)
    choosen_count = len(denominator_sentences)
    assert choosen_count != 0
    return common_count / choosen_count