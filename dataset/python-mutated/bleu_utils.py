"""

TODO: the code is take from Apache-2 Licensed NLTK: make sure we do this properly!


Copied over from nltk.tranlate.bleu_score. This code has two major changes:
 - allows to turn off length/brevity penalty --- it has no sense for self-bleu,
 - allows to use arithmetic instead of geometric mean
"""
import math
import sys
from fractions import Fraction
import warnings
from collections import Counter
from nltk.translate.bleu_score import modified_precision, closest_ref_length, brevity_penalty, SmoothingFunction

def corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=None, auto_reweigh=False, averaging_mode='geometric', no_length_penalty=False):
    if False:
        i = 10
        return i + 15
    "\n    Calculate a single corpus-level BLEU score (aka. system-level BLEU) for all\n    the hypotheses and their respective references.\n\n    Instead of averaging the sentence level BLEU scores (i.e. marco-average\n    precision), the original BLEU metric (Papineni et al. 2002) accounts for\n    the micro-average precision (i.e. summing the numerators and denominators\n    for each hypothesis-reference(s) pairs before the division).\n\n    >>> hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',\n    ...         'ensures', 'that', 'the', 'military', 'always',\n    ...         'obeys', 'the', 'commands', 'of', 'the', 'party']\n    >>> ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',\n    ...          'ensures', 'that', 'the', 'military', 'will', 'forever',\n    ...          'heed', 'Party', 'commands']\n    >>> ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which',\n    ...          'guarantees', 'the', 'military', 'forces', 'always',\n    ...          'being', 'under', 'the', 'command', 'of', 'the', 'Party']\n    >>> ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',\n    ...          'army', 'always', 'to', 'heed', 'the', 'directions',\n    ...          'of', 'the', 'party']\n\n    >>> hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',\n    ...         'interested', 'in', 'world', 'history']\n    >>> ref2a = ['he', 'was', 'interested', 'in', 'world', 'history',\n    ...          'because', 'he', 'read', 'the', 'book']\n\n    >>> list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]\n    >>> hypotheses = [hyp1, hyp2]\n    >>> corpus_bleu(list_of_references, hypotheses) # doctest: +ELLIPSIS\n    0.5920...\n\n    The example below show that corpus_bleu() is different from averaging\n    sentence_bleu() for hypotheses\n\n    >>> score1 = sentence_bleu([ref1a, ref1b, ref1c], hyp1)\n    >>> score2 = sentence_bleu([ref2a], hyp2)\n    >>> (score1 + score2) / 2 # doctest: +ELLIPSIS\n    0.6223...\n\n    :param list_of_references: a corpus of lists of reference sentences, w.r.t. hypotheses\n    :type list_of_references: list(list(list(str)))\n    :param hypotheses: a list of hypothesis sentences\n    :type hypotheses: list(list(str))\n    :param weights: weights for unigrams, bigrams, trigrams and so on\n    :type weights: list(float)\n    :param smoothing_function:\n    :type smoothing_function: SmoothingFunction\n    :param auto_reweigh: Option to re-normalize the weights uniformly.\n    :type auto_reweigh: bool\n    :return: The corpus-level BLEU score.\n    :rtype: float\n    "
    p_numerators = Counter()
    p_denominators = Counter()
    (hyp_lengths, ref_lengths) = (0, 0)
    assert len(list_of_references) == len(hypotheses), 'The number of hypotheses and their reference(s) should be the same '
    for (references, hypothesis) in zip(list_of_references, hypotheses):
        for (i, _) in enumerate(weights, start=1):
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator
        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)
    if no_length_penalty and averaging_mode == 'geometric':
        bp = 1.0
    elif no_length_penalty and averaging_mode == 'arithmetic':
        bp = 0.0
    else:
        assert not no_length_penalty
        assert averaging_mode != 'arithmetic', 'Not sure how to apply length penalty when aurithmetic mode'
        bp = brevity_penalty(ref_lengths, hyp_lengths)
    if auto_reweigh:
        if hyp_lengths < 4 and weights == (0.25, 0.25, 0.25, 0.25):
            weights = (1 / hyp_lengths,) * hyp_lengths
    p_n = [Fraction(p_numerators[i], p_denominators[i], _normalize=False) for (i, _) in enumerate(weights, start=1)]
    if p_numerators[1] == 0:
        return 0
    if not smoothing_function:
        smoothing_function = SmoothingFunction().method0
    p_n = smoothing_function(p_n, references=references, hypothesis=hypothesis, hyp_len=hyp_lengths)
    if averaging_mode == 'geometric':
        s = (w_i * math.log(p_i) for (w_i, p_i) in zip(weights, p_n))
        s = bp * math.exp(math.fsum(s))
    elif averaging_mode == 'arithmetic':
        s = (w_i * p_i for (w_i, p_i) in zip(weights, p_n))
        s = math.fsum(s)
    return s

def sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=None, auto_reweigh=False, averaging_mode='geometric', no_length_penalty=False):
    if False:
        for i in range(10):
            print('nop')
    return corpus_bleu([references], [hypothesis], weights, smoothing_function, auto_reweigh, averaging_mode, no_length_penalty)