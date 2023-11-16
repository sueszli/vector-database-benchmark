"""BLEU score implementation."""
import math
import sys
import warnings
from collections import Counter
from fractions import Fraction
from nltk.util import ngrams

def sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=None, auto_reweigh=False):
    if False:
        print('Hello World!')
    '\n    Calculate BLEU score (Bilingual Evaluation Understudy) from\n    Papineni, Kishore, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002.\n    "BLEU: a method for automatic evaluation of machine translation."\n    In Proceedings of ACL. https://www.aclweb.org/anthology/P02-1040.pdf\n\n    >>> hypothesis1 = [\'It\', \'is\', \'a\', \'guide\', \'to\', \'action\', \'which\',\n    ...               \'ensures\', \'that\', \'the\', \'military\', \'always\',\n    ...               \'obeys\', \'the\', \'commands\', \'of\', \'the\', \'party\']\n\n    >>> hypothesis2 = [\'It\', \'is\', \'to\', \'insure\', \'the\', \'troops\',\n    ...               \'forever\', \'hearing\', \'the\', \'activity\', \'guidebook\',\n    ...               \'that\', \'party\', \'direct\']\n\n    >>> reference1 = [\'It\', \'is\', \'a\', \'guide\', \'to\', \'action\', \'that\',\n    ...               \'ensures\', \'that\', \'the\', \'military\', \'will\', \'forever\',\n    ...               \'heed\', \'Party\', \'commands\']\n\n    >>> reference2 = [\'It\', \'is\', \'the\', \'guiding\', \'principle\', \'which\',\n    ...               \'guarantees\', \'the\', \'military\', \'forces\', \'always\',\n    ...               \'being\', \'under\', \'the\', \'command\', \'of\', \'the\',\n    ...               \'Party\']\n\n    >>> reference3 = [\'It\', \'is\', \'the\', \'practical\', \'guide\', \'for\', \'the\',\n    ...               \'army\', \'always\', \'to\', \'heed\', \'the\', \'directions\',\n    ...               \'of\', \'the\', \'party\']\n\n    >>> sentence_bleu([reference1, reference2, reference3], hypothesis1) # doctest: +ELLIPSIS\n    0.5045...\n\n    If there is no ngrams overlap for any order of n-grams, BLEU returns the\n    value 0. This is because the precision for the order of n-grams without\n    overlap is 0, and the geometric mean in the final BLEU score computation\n    multiplies the 0 with the precision of other n-grams. This results in 0\n    (independently of the precision of the other n-gram orders). The following\n    example has zero 3-gram and 4-gram overlaps:\n\n    >>> round(sentence_bleu([reference1, reference2, reference3], hypothesis2),4) # doctest: +ELLIPSIS\n    0.0\n\n    To avoid this harsh behaviour when no ngram overlaps are found a smoothing\n    function can be used.\n\n    >>> chencherry = SmoothingFunction()\n    >>> sentence_bleu([reference1, reference2, reference3], hypothesis2,\n    ...     smoothing_function=chencherry.method1) # doctest: +ELLIPSIS\n    0.0370...\n\n    The default BLEU calculates a score for up to 4-grams using uniform\n    weights (this is called BLEU-4). To evaluate your translations with\n    higher/lower order ngrams, use customized weights. E.g. when accounting\n    for up to 5-grams with uniform weights (this is called BLEU-5) use:\n\n    >>> weights = (1./5., 1./5., 1./5., 1./5., 1./5.)\n    >>> sentence_bleu([reference1, reference2, reference3], hypothesis1, weights) # doctest: +ELLIPSIS\n    0.3920...\n\n    Multiple BLEU scores can be computed at once, by supplying a list of weights.\n    E.g. for computing BLEU-2, BLEU-3 *and* BLEU-4 in one computation, use:\n    >>> weights = [\n    ...     (1./2., 1./2.),\n    ...     (1./3., 1./3., 1./3.),\n    ...     (1./4., 1./4., 1./4., 1./4.)\n    ... ]\n    >>> sentence_bleu([reference1, reference2, reference3], hypothesis1, weights) # doctest: +ELLIPSIS\n    [0.7453..., 0.6240..., 0.5045...]\n\n    :param references: reference sentences\n    :type references: list(list(str))\n    :param hypothesis: a hypothesis sentence\n    :type hypothesis: list(str)\n    :param weights: weights for unigrams, bigrams, trigrams and so on (one or a list of weights)\n    :type weights: tuple(float) / list(tuple(float))\n    :param smoothing_function:\n    :type smoothing_function: SmoothingFunction\n    :param auto_reweigh: Option to re-normalize the weights uniformly.\n    :type auto_reweigh: bool\n    :return: The sentence-level BLEU score. Returns a list if multiple weights were supplied.\n    :rtype: float / list(float)\n    '
    return corpus_bleu([references], [hypothesis], weights, smoothing_function, auto_reweigh)

def corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=None, auto_reweigh=False):
    if False:
        while True:
            i = 10
    "\n    Calculate a single corpus-level BLEU score (aka. system-level BLEU) for all\n    the hypotheses and their respective references.\n\n    Instead of averaging the sentence level BLEU scores (i.e. macro-average\n    precision), the original BLEU metric (Papineni et al. 2002) accounts for\n    the micro-average precision (i.e. summing the numerators and denominators\n    for each hypothesis-reference(s) pairs before the division).\n\n    >>> hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which',\n    ...         'ensures', 'that', 'the', 'military', 'always',\n    ...         'obeys', 'the', 'commands', 'of', 'the', 'party']\n    >>> ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that',\n    ...          'ensures', 'that', 'the', 'military', 'will', 'forever',\n    ...          'heed', 'Party', 'commands']\n    >>> ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which',\n    ...          'guarantees', 'the', 'military', 'forces', 'always',\n    ...          'being', 'under', 'the', 'command', 'of', 'the', 'Party']\n    >>> ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the',\n    ...          'army', 'always', 'to', 'heed', 'the', 'directions',\n    ...          'of', 'the', 'party']\n\n    >>> hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was',\n    ...         'interested', 'in', 'world', 'history']\n    >>> ref2a = ['he', 'was', 'interested', 'in', 'world', 'history',\n    ...          'because', 'he', 'read', 'the', 'book']\n\n    >>> list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]\n    >>> hypotheses = [hyp1, hyp2]\n    >>> corpus_bleu(list_of_references, hypotheses) # doctest: +ELLIPSIS\n    0.5920...\n\n    The example below show that corpus_bleu() is different from averaging\n    sentence_bleu() for hypotheses\n\n    >>> score1 = sentence_bleu([ref1a, ref1b, ref1c], hyp1)\n    >>> score2 = sentence_bleu([ref2a], hyp2)\n    >>> (score1 + score2) / 2 # doctest: +ELLIPSIS\n    0.6223...\n\n    Custom weights may be supplied to fine-tune the BLEU score further.\n    A tuple of float weights for unigrams, bigrams, trigrams and so on can be given.\n    >>> weights = (0.1, 0.3, 0.5, 0.1)\n    >>> corpus_bleu(list_of_references, hypotheses, weights=weights) # doctest: +ELLIPSIS\n    0.5818...\n\n    This particular weight gave extra value to trigrams.\n    Furthermore, multiple weights can be given, resulting in multiple BLEU scores.\n    >>> weights = [\n    ...     (0.5, 0.5),\n    ...     (0.333, 0.333, 0.334),\n    ...     (0.25, 0.25, 0.25, 0.25),\n    ...     (0.2, 0.2, 0.2, 0.2, 0.2)\n    ... ]\n    >>> corpus_bleu(list_of_references, hypotheses, weights=weights) # doctest: +ELLIPSIS\n    [0.8242..., 0.7067..., 0.5920..., 0.4719...]\n\n    :param list_of_references: a corpus of lists of reference sentences, w.r.t. hypotheses\n    :type list_of_references: list(list(list(str)))\n    :param hypotheses: a list of hypothesis sentences\n    :type hypotheses: list(list(str))\n    :param weights: weights for unigrams, bigrams, trigrams and so on (one or a list of weights)\n    :type weights: tuple(float) / list(tuple(float))\n    :param smoothing_function:\n    :type smoothing_function: SmoothingFunction\n    :param auto_reweigh: Option to re-normalize the weights uniformly.\n    :type auto_reweigh: bool\n    :return: The corpus-level BLEU score.\n    :rtype: float\n    "
    p_numerators = Counter()
    p_denominators = Counter()
    (hyp_lengths, ref_lengths) = (0, 0)
    assert len(list_of_references) == len(hypotheses), 'The number of hypotheses and their reference(s) should be the same '
    try:
        weights[0][0]
    except:
        weights = [weights]
    max_weight_length = max((len(weight) for weight in weights))
    for (references, hypothesis) in zip(list_of_references, hypotheses):
        for i in range(1, max_weight_length + 1):
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator
        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)
    bp = brevity_penalty(ref_lengths, hyp_lengths)
    p_n = [Fraction(p_numerators[i], p_denominators[i], _normalize=False) for i in range(1, max_weight_length + 1)]
    if p_numerators[1] == 0:
        return 0 if len(weights) == 1 else [0] * len(weights)
    if not smoothing_function:
        smoothing_function = SmoothingFunction().method0
    p_n = smoothing_function(p_n, references=references, hypothesis=hypothesis, hyp_len=hyp_lengths)
    bleu_scores = []
    for weight in weights:
        if auto_reweigh:
            if hyp_lengths < 4 and weight == (0.25, 0.25, 0.25, 0.25):
                weight = (1 / hyp_lengths,) * hyp_lengths
        s = (w_i * math.log(p_i) for (w_i, p_i) in zip(weight, p_n) if p_i > 0)
        s = bp * math.exp(math.fsum(s))
        bleu_scores.append(s)
    return bleu_scores[0] if len(weights) == 1 else bleu_scores

def modified_precision(references, hypothesis, n):
    if False:
        print('Hello World!')
    '\n    Calculate modified ngram precision.\n\n    The normal precision method may lead to some wrong translations with\n    high-precision, e.g., the translation, in which a word of reference\n    repeats several times, has very high precision.\n\n    This function only returns the Fraction object that contains the numerator\n    and denominator necessary to calculate the corpus-level precision.\n    To calculate the modified precision for a single pair of hypothesis and\n    references, cast the Fraction object into a float.\n\n    The famous "the the the ... " example shows that you can get BLEU precision\n    by duplicating high frequency words.\n\n        >>> reference1 = \'the cat is on the mat\'.split()\n        >>> reference2 = \'there is a cat on the mat\'.split()\n        >>> hypothesis1 = \'the the the the the the the\'.split()\n        >>> references = [reference1, reference2]\n        >>> float(modified_precision(references, hypothesis1, n=1)) # doctest: +ELLIPSIS\n        0.2857...\n\n    In the modified n-gram precision, a reference word will be considered\n    exhausted after a matching hypothesis word is identified, e.g.\n\n        >>> reference1 = [\'It\', \'is\', \'a\', \'guide\', \'to\', \'action\', \'that\',\n        ...               \'ensures\', \'that\', \'the\', \'military\', \'will\',\n        ...               \'forever\', \'heed\', \'Party\', \'commands\']\n        >>> reference2 = [\'It\', \'is\', \'the\', \'guiding\', \'principle\', \'which\',\n        ...               \'guarantees\', \'the\', \'military\', \'forces\', \'always\',\n        ...               \'being\', \'under\', \'the\', \'command\', \'of\', \'the\',\n        ...               \'Party\']\n        >>> reference3 = [\'It\', \'is\', \'the\', \'practical\', \'guide\', \'for\', \'the\',\n        ...               \'army\', \'always\', \'to\', \'heed\', \'the\', \'directions\',\n        ...               \'of\', \'the\', \'party\']\n        >>> hypothesis = \'of the\'.split()\n        >>> references = [reference1, reference2, reference3]\n        >>> float(modified_precision(references, hypothesis, n=1))\n        1.0\n        >>> float(modified_precision(references, hypothesis, n=2))\n        1.0\n\n    An example of a normal machine translation hypothesis:\n\n        >>> hypothesis1 = [\'It\', \'is\', \'a\', \'guide\', \'to\', \'action\', \'which\',\n        ...               \'ensures\', \'that\', \'the\', \'military\', \'always\',\n        ...               \'obeys\', \'the\', \'commands\', \'of\', \'the\', \'party\']\n\n        >>> hypothesis2 = [\'It\', \'is\', \'to\', \'insure\', \'the\', \'troops\',\n        ...               \'forever\', \'hearing\', \'the\', \'activity\', \'guidebook\',\n        ...               \'that\', \'party\', \'direct\']\n\n        >>> reference1 = [\'It\', \'is\', \'a\', \'guide\', \'to\', \'action\', \'that\',\n        ...               \'ensures\', \'that\', \'the\', \'military\', \'will\',\n        ...               \'forever\', \'heed\', \'Party\', \'commands\']\n\n        >>> reference2 = [\'It\', \'is\', \'the\', \'guiding\', \'principle\', \'which\',\n        ...               \'guarantees\', \'the\', \'military\', \'forces\', \'always\',\n        ...               \'being\', \'under\', \'the\', \'command\', \'of\', \'the\',\n        ...               \'Party\']\n\n        >>> reference3 = [\'It\', \'is\', \'the\', \'practical\', \'guide\', \'for\', \'the\',\n        ...               \'army\', \'always\', \'to\', \'heed\', \'the\', \'directions\',\n        ...               \'of\', \'the\', \'party\']\n        >>> references = [reference1, reference2, reference3]\n        >>> float(modified_precision(references, hypothesis1, n=1)) # doctest: +ELLIPSIS\n        0.9444...\n        >>> float(modified_precision(references, hypothesis2, n=1)) # doctest: +ELLIPSIS\n        0.5714...\n        >>> float(modified_precision(references, hypothesis1, n=2)) # doctest: +ELLIPSIS\n        0.5882352941176471\n        >>> float(modified_precision(references, hypothesis2, n=2)) # doctest: +ELLIPSIS\n        0.07692...\n\n\n    :param references: A list of reference translations.\n    :type references: list(list(str))\n    :param hypothesis: A hypothesis translation.\n    :type hypothesis: list(str)\n    :param n: The ngram order.\n    :type n: int\n    :return: BLEU\'s modified precision for the nth order ngram.\n    :rtype: Fraction\n    '
    counts = Counter(ngrams(hypothesis, n)) if len(hypothesis) >= n else Counter()
    max_counts = {}
    for reference in references:
        reference_counts = Counter(ngrams(reference, n)) if len(reference) >= n else Counter()
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])
    clipped_counts = {ngram: min(count, max_counts[ngram]) for (ngram, count) in counts.items()}
    numerator = sum(clipped_counts.values())
    denominator = max(1, sum(counts.values()))
    return Fraction(numerator, denominator, _normalize=False)

def closest_ref_length(references, hyp_len):
    if False:
        print('Hello World!')
    "\n    This function finds the reference that is the closest length to the\n    hypothesis. The closest reference length is referred to as *r* variable\n    from the brevity penalty formula in Papineni et. al. (2002)\n\n    :param references: A list of reference translations.\n    :type references: list(list(str))\n    :param hyp_len: The length of the hypothesis.\n    :type hyp_len: int\n    :return: The length of the reference that's closest to the hypothesis.\n    :rtype: int\n    "
    ref_lens = (len(reference) for reference in references)
    closest_ref_len = min(ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len))
    return closest_ref_len

def brevity_penalty(closest_ref_len, hyp_len):
    if False:
        print('Hello World!')
    "\n    Calculate brevity penalty.\n\n    As the modified n-gram precision still has the problem from the short\n    length sentence, brevity penalty is used to modify the overall BLEU\n    score according to length.\n\n    An example from the paper. There are three references with length 12, 15\n    and 17. And a concise hypothesis of the length 12. The brevity penalty is 1.\n\n    >>> reference1 = list('aaaaaaaaaaaa')      # i.e. ['a'] * 12\n    >>> reference2 = list('aaaaaaaaaaaaaaa')   # i.e. ['a'] * 15\n    >>> reference3 = list('aaaaaaaaaaaaaaaaa') # i.e. ['a'] * 17\n    >>> hypothesis = list('aaaaaaaaaaaa')      # i.e. ['a'] * 12\n    >>> references = [reference1, reference2, reference3]\n    >>> hyp_len = len(hypothesis)\n    >>> closest_ref_len =  closest_ref_length(references, hyp_len)\n    >>> brevity_penalty(closest_ref_len, hyp_len)\n    1.0\n\n    In case a hypothesis translation is shorter than the references, penalty is\n    applied.\n\n    >>> references = [['a'] * 28, ['a'] * 28]\n    >>> hypothesis = ['a'] * 12\n    >>> hyp_len = len(hypothesis)\n    >>> closest_ref_len =  closest_ref_length(references, hyp_len)\n    >>> brevity_penalty(closest_ref_len, hyp_len)\n    0.2635971381157267\n\n    The length of the closest reference is used to compute the penalty. If the\n    length of a hypothesis is 12, and the reference lengths are 13 and 2, the\n    penalty is applied because the hypothesis length (12) is less then the\n    closest reference length (13).\n\n    >>> references = [['a'] * 13, ['a'] * 2]\n    >>> hypothesis = ['a'] * 12\n    >>> hyp_len = len(hypothesis)\n    >>> closest_ref_len =  closest_ref_length(references, hyp_len)\n    >>> brevity_penalty(closest_ref_len, hyp_len) # doctest: +ELLIPSIS\n    0.9200...\n\n    The brevity penalty doesn't depend on reference order. More importantly,\n    when two reference sentences are at the same distance, the shortest\n    reference sentence length is used.\n\n    >>> references = [['a'] * 13, ['a'] * 11]\n    >>> hypothesis = ['a'] * 12\n    >>> hyp_len = len(hypothesis)\n    >>> closest_ref_len =  closest_ref_length(references, hyp_len)\n    >>> bp1 = brevity_penalty(closest_ref_len, hyp_len)\n    >>> hyp_len = len(hypothesis)\n    >>> closest_ref_len =  closest_ref_length(reversed(references), hyp_len)\n    >>> bp2 = brevity_penalty(closest_ref_len, hyp_len)\n    >>> bp1 == bp2 == 1\n    True\n\n    A test example from mteval-v13a.pl (starting from the line 705):\n\n    >>> references = [['a'] * 11, ['a'] * 8]\n    >>> hypothesis = ['a'] * 7\n    >>> hyp_len = len(hypothesis)\n    >>> closest_ref_len =  closest_ref_length(references, hyp_len)\n    >>> brevity_penalty(closest_ref_len, hyp_len) # doctest: +ELLIPSIS\n    0.8668...\n\n    >>> references = [['a'] * 11, ['a'] * 8, ['a'] * 6, ['a'] * 7]\n    >>> hypothesis = ['a'] * 7\n    >>> hyp_len = len(hypothesis)\n    >>> closest_ref_len =  closest_ref_length(references, hyp_len)\n    >>> brevity_penalty(closest_ref_len, hyp_len)\n    1.0\n\n    :param hyp_len: The length of the hypothesis for a single sentence OR the\n        sum of all the hypotheses' lengths for a corpus\n    :type hyp_len: int\n    :param closest_ref_len: The length of the closest reference for a single\n        hypothesis OR the sum of all the closest references for every hypotheses.\n    :type closest_ref_len: int\n    :return: BLEU's brevity penalty.\n    :rtype: float\n    "
    if hyp_len > closest_ref_len:
        return 1
    elif hyp_len == 0:
        return 0
    else:
        return math.exp(1 - closest_ref_len / hyp_len)

class SmoothingFunction:
    """
    This is an implementation of the smoothing techniques
    for segment-level BLEU scores that was presented in
    Boxing Chen and Collin Cherry (2014) A Systematic Comparison of
    Smoothing Techniques for Sentence-Level BLEU. In WMT14.
    http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf
    """

    def __init__(self, epsilon=0.1, alpha=5, k=5):
        if False:
            for i in range(10):
                print('nop')
        "\n        This will initialize the parameters required for the various smoothing\n        techniques, the default values are set to the numbers used in the\n        experiments from Chen and Cherry (2014).\n\n        >>> hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures',\n        ...                 'that', 'the', 'military', 'always', 'obeys', 'the',\n        ...                 'commands', 'of', 'the', 'party']\n        >>> reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures',\n        ...               'that', 'the', 'military', 'will', 'forever', 'heed',\n        ...               'Party', 'commands']\n\n        >>> chencherry = SmoothingFunction()\n        >>> print(sentence_bleu([reference1], hypothesis1)) # doctest: +ELLIPSIS\n        0.4118...\n        >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method0)) # doctest: +ELLIPSIS\n        0.4118...\n        >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method1)) # doctest: +ELLIPSIS\n        0.4118...\n        >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method2)) # doctest: +ELLIPSIS\n        0.4452...\n        >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method3)) # doctest: +ELLIPSIS\n        0.4118...\n        >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method4)) # doctest: +ELLIPSIS\n        0.4118...\n        >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method5)) # doctest: +ELLIPSIS\n        0.4905...\n        >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method6)) # doctest: +ELLIPSIS\n        0.4135...\n        >>> print(sentence_bleu([reference1], hypothesis1, smoothing_function=chencherry.method7)) # doctest: +ELLIPSIS\n        0.4905...\n\n        :param epsilon: the epsilon value use in method 1\n        :type epsilon: float\n        :param alpha: the alpha value use in method 6\n        :type alpha: int\n        :param k: the k value use in method 4\n        :type k: int\n        "
        self.epsilon = epsilon
        self.alpha = alpha
        self.k = k

    def method0(self, p_n, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        No smoothing.\n        '
        p_n_new = []
        for (i, p_i) in enumerate(p_n):
            if p_i.numerator != 0:
                p_n_new.append(p_i)
            else:
                _msg = str('\nThe hypothesis contains 0 counts of {}-gram overlaps.\nTherefore the BLEU score evaluates to 0, independently of\nhow many N-gram overlaps of lower order it contains.\nConsider using lower n-gram order or use SmoothingFunction()').format(i + 1)
                warnings.warn(_msg)
                p_n_new.append(sys.float_info.min)
        return p_n_new

    def method1(self, p_n, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Smoothing method 1: Add *epsilon* counts to precision with 0 counts.\n        '
        return [(p_i.numerator + self.epsilon) / p_i.denominator if p_i.numerator == 0 else p_i for p_i in p_n]

    def method2(self, p_n, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Smoothing method 2: Add 1 to both numerator and denominator from\n        Chin-Yew Lin and Franz Josef Och (2004) ORANGE: a Method for\n        Evaluating Automatic Evaluation Metrics for Machine Translation.\n        In COLING 2004.\n        '
        return [Fraction(p_n[i].numerator + 1, p_n[i].denominator + 1, _normalize=False) if i != 0 else p_n[0] for i in range(len(p_n))]

    def method3(self, p_n, *args, **kwargs):
        if False:
            print('Hello World!')
        "\n        Smoothing method 3: NIST geometric sequence smoothing\n        The smoothing is computed by taking 1 / ( 2^k ), instead of 0, for each\n        precision score whose matching n-gram count is null.\n        k is 1 for the first 'n' value for which the n-gram match count is null/\n\n        For example, if the text contains:\n\n        - one 2-gram match\n        - and (consequently) two 1-gram matches\n\n        the n-gram count for each individual precision score would be:\n\n        - n=1  =>  prec_count = 2     (two unigrams)\n        - n=2  =>  prec_count = 1     (one bigram)\n        - n=3  =>  prec_count = 1/2   (no trigram,  taking 'smoothed' value of 1 / ( 2^k ), with k=1)\n        - n=4  =>  prec_count = 1/4   (no fourgram, taking 'smoothed' value of 1 / ( 2^k ), with k=2)\n        "
        incvnt = 1
        for (i, p_i) in enumerate(p_n):
            if p_i.numerator == 0:
                p_n[i] = 1 / (2 ** incvnt * p_i.denominator)
                incvnt += 1
        return p_n

    def method4(self, p_n, references, hypothesis, hyp_len=None, *args, **kwargs):
        if False:
            return 10
        '\n        Smoothing method 4:\n        Shorter translations may have inflated precision values due to having\n        smaller denominators; therefore, we give them proportionally\n        smaller smoothed counts. Instead of scaling to 1/(2^k), Chen and Cherry\n        suggests dividing by 1/ln(len(T)), where T is the length of the translation.\n        '
        incvnt = 1
        hyp_len = hyp_len if hyp_len else len(hypothesis)
        for (i, p_i) in enumerate(p_n):
            if p_i.numerator == 0 and hyp_len > 1:
                numerator = 1 / (2 ** incvnt * self.k / math.log(hyp_len))
                p_n[i] = numerator / p_i.denominator
                incvnt += 1
        return p_n

    def method5(self, p_n, references, hypothesis, hyp_len=None, *args, **kwargs):
        if False:
            return 10
        '\n        Smoothing method 5:\n        The matched counts for similar values of n should be similar. To a\n        calculate the n-gram matched count, it averages the n−1, n and n+1 gram\n        matched counts.\n        '
        hyp_len = hyp_len if hyp_len else len(hypothesis)
        m = {}
        p_n_plus1 = p_n + [modified_precision(references, hypothesis, 5)]
        m[-1] = p_n[0] + 1
        for (i, p_i) in enumerate(p_n):
            p_n[i] = (m[i - 1] + p_i + p_n_plus1[i + 1]) / 3
            m[i] = p_n[i]
        return p_n

    def method6(self, p_n, references, hypothesis, hyp_len=None, *args, **kwargs):
        if False:
            return 10
        '\n        Smoothing method 6:\n        Interpolates the maximum likelihood estimate of the precision *p_n* with\n        a prior estimate *pi0*. The prior is estimated by assuming that the ratio\n        between pn and pn−1 will be the same as that between pn−1 and pn−2; from\n        Gao and He (2013) Training MRF-Based Phrase Translation Models using\n        Gradient Ascent. In NAACL.\n        '
        hyp_len = hyp_len if hyp_len else len(hypothesis)
        assert p_n[2], 'This smoothing method requires non-zero precision for bigrams.'
        for (i, p_i) in enumerate(p_n):
            if i in [0, 1]:
                continue
            else:
                pi0 = 0 if p_n[i - 2] == 0 else p_n[i - 1] ** 2 / p_n[i - 2]
                m = p_i.numerator
                l = sum((1 for _ in ngrams(hypothesis, i + 1)))
                p_n[i] = (m + self.alpha * pi0) / (l + self.alpha)
        return p_n

    def method7(self, p_n, references, hypothesis, hyp_len=None, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Smoothing method 7:\n        Interpolates methods 4 and 5.\n        '
        hyp_len = hyp_len if hyp_len else len(hypothesis)
        p_n = self.method4(p_n, references, hypothesis, hyp_len)
        p_n = self.method5(p_n, references, hypothesis, hyp_len)
        return p_n