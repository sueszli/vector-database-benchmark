"""NIST score implementation."""
import fractions
import math
from collections import Counter
from nltk.util import ngrams

def sentence_nist(references, hypothesis, n=5):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate NIST score from\n    George Doddington. 2002. "Automatic evaluation of machine translation quality\n    using n-gram co-occurrence statistics." Proceedings of HLT.\n    Morgan Kaufmann Publishers Inc. https://dl.acm.org/citation.cfm?id=1289189.1289273\n\n    DARPA commissioned NIST to develop an MT evaluation facility based on the BLEU\n    score. The official script used by NIST to compute BLEU and NIST score is\n    mteval-14.pl. The main differences are:\n\n     - BLEU uses geometric mean of the ngram overlaps, NIST uses arithmetic mean.\n     - NIST has a different brevity penalty\n     - NIST score from mteval-14.pl has a self-contained tokenizer\n\n    Note: The mteval-14.pl includes a smoothing function for BLEU score that is NOT\n          used in the NIST score computation.\n\n    >>> hypothesis1 = [\'It\', \'is\', \'a\', \'guide\', \'to\', \'action\', \'which\',\n    ...               \'ensures\', \'that\', \'the\', \'military\', \'always\',\n    ...               \'obeys\', \'the\', \'commands\', \'of\', \'the\', \'party\']\n\n    >>> hypothesis2 = [\'It\', \'is\', \'to\', \'insure\', \'the\', \'troops\',\n    ...               \'forever\', \'hearing\', \'the\', \'activity\', \'guidebook\',\n    ...               \'that\', \'party\', \'direct\']\n\n    >>> reference1 = [\'It\', \'is\', \'a\', \'guide\', \'to\', \'action\', \'that\',\n    ...               \'ensures\', \'that\', \'the\', \'military\', \'will\', \'forever\',\n    ...               \'heed\', \'Party\', \'commands\']\n\n    >>> reference2 = [\'It\', \'is\', \'the\', \'guiding\', \'principle\', \'which\',\n    ...               \'guarantees\', \'the\', \'military\', \'forces\', \'always\',\n    ...               \'being\', \'under\', \'the\', \'command\', \'of\', \'the\',\n    ...               \'Party\']\n\n    >>> reference3 = [\'It\', \'is\', \'the\', \'practical\', \'guide\', \'for\', \'the\',\n    ...               \'army\', \'always\', \'to\', \'heed\', \'the\', \'directions\',\n    ...               \'of\', \'the\', \'party\']\n\n    >>> sentence_nist([reference1, reference2, reference3], hypothesis1) # doctest: +ELLIPSIS\n    3.3709...\n\n    >>> sentence_nist([reference1, reference2, reference3], hypothesis2) # doctest: +ELLIPSIS\n    1.4619...\n\n    :param references: reference sentences\n    :type references: list(list(str))\n    :param hypothesis: a hypothesis sentence\n    :type hypothesis: list(str)\n    :param n: highest n-gram order\n    :type n: int\n    '
    return corpus_nist([references], [hypothesis], n)

def corpus_nist(list_of_references, hypotheses, n=5):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate a single corpus-level NIST score (aka. system-level BLEU) for all\n    the hypotheses and their respective references.\n\n    :param references: a corpus of lists of reference sentences, w.r.t. hypotheses\n    :type references: list(list(list(str)))\n    :param hypotheses: a list of hypothesis sentences\n    :type hypotheses: list(list(str))\n    :param n: highest n-gram order\n    :type n: int\n    '
    assert len(list_of_references) == len(hypotheses), 'The number of hypotheses and their reference(s) should be the same'
    ngram_freq = Counter()
    total_reference_words = 0
    for references in list_of_references:
        for reference in references:
            for i in range(1, n + 1):
                ngram_freq.update(ngrams(reference, i))
            total_reference_words += len(reference)
    information_weights = {}
    for _ngram in ngram_freq:
        _mgram = _ngram[:-1]
        if _mgram and _mgram in ngram_freq:
            numerator = ngram_freq[_mgram]
        else:
            numerator = total_reference_words
        information_weights[_ngram] = math.log(numerator / ngram_freq[_ngram], 2)
    nist_precision_numerator_per_ngram = Counter()
    nist_precision_denominator_per_ngram = Counter()
    (l_ref, l_sys) = (0, 0)
    for i in range(1, n + 1):
        for (references, hypothesis) in zip(list_of_references, hypotheses):
            hyp_len = len(hypothesis)
            nist_score_per_ref = []
            for reference in references:
                _ref_len = len(reference)
                hyp_ngrams = Counter(ngrams(hypothesis, i)) if len(hypothesis) >= i else Counter()
                ref_ngrams = Counter(ngrams(reference, i)) if len(reference) >= i else Counter()
                ngram_overlaps = hyp_ngrams & ref_ngrams
                _numerator = sum((information_weights[_ngram] * count for (_ngram, count) in ngram_overlaps.items()))
                _denominator = sum(hyp_ngrams.values())
                _precision = 0 if _denominator == 0 else _numerator / _denominator
                nist_score_per_ref.append((_precision, _numerator, _denominator, _ref_len))
            (precision, numerator, denominator, ref_len) = max(nist_score_per_ref)
            nist_precision_numerator_per_ngram[i] += numerator
            nist_precision_denominator_per_ngram[i] += denominator
            l_ref += ref_len
            l_sys += hyp_len
    nist_precision = 0
    for i in nist_precision_numerator_per_ngram:
        precision = nist_precision_numerator_per_ngram[i] / nist_precision_denominator_per_ngram[i]
        nist_precision += precision
    return nist_precision * nist_length_penalty(l_ref, l_sys)

def nist_length_penalty(ref_len, hyp_len):
    if False:
        for i in range(10):
            print('nop')
    "\n    Calculates the NIST length penalty, from Eq. 3 in Doddington (2002)\n\n        penalty = exp( beta * log( min( len(hyp)/len(ref) , 1.0 )))\n\n    where,\n\n        `beta` is chosen to make the brevity penalty factor = 0.5 when the\n        no. of words in the system output (hyp) is 2/3 of the average\n        no. of words in the reference translation (ref)\n\n    The NIST penalty is different from BLEU's such that it minimize the impact\n    of the score of small variations in the length of a translation.\n    See Fig. 4 in  Doddington (2002)\n    "
    ratio = hyp_len / ref_len
    if 0 < ratio < 1:
        (ratio_x, score_x) = (1.5, 0.5)
        beta = math.log(score_x) / math.log(ratio_x) ** 2
        return math.exp(beta * math.log(ratio) ** 2)
    else:
        return max(min(ratio, 1.0), 0.0)