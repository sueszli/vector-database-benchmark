""" ChrF score implementation """
import re
from collections import Counter, defaultdict
from nltk.util import ngrams

def sentence_chrf(reference, hypothesis, min_len=1, max_len=6, beta=3.0, ignore_whitespace=True):
    if False:
        i = 10
        return i + 15
    '\n    Calculates the sentence level CHRF (Character n-gram F-score) described in\n     - Maja Popovic. 2015. CHRF: Character n-gram F-score for Automatic MT Evaluation.\n       In Proceedings of the 10th Workshop on Machine Translation.\n       https://www.statmt.org/wmt15/pdf/WMT49.pdf\n     - Maja Popovic. 2016. CHRF Deconstructed: β Parameters and n-gram Weights.\n       In Proceedings of the 1st Conference on Machine Translation.\n       https://www.statmt.org/wmt16/pdf/W16-2341.pdf\n\n    This implementation of CHRF only supports a single reference at the moment.\n\n    For details not reported in the paper, consult Maja Popovic\'s original\n    implementation: https://github.com/m-popovic/chrF\n\n    The code should output results equivalent to running CHRF++ with the\n    following options: -nw 0 -b 3\n\n    An example from the original BLEU paper\n    https://www.aclweb.org/anthology/P02-1040.pdf\n\n        >>> ref1 = str(\'It is a guide to action that ensures that the military \'\n        ...            \'will forever heed Party commands\').split()\n        >>> hyp1 = str(\'It is a guide to action which ensures that the military \'\n        ...            \'always obeys the commands of the party\').split()\n        >>> hyp2 = str(\'It is to insure the troops forever hearing the activity \'\n        ...            \'guidebook that party direct\').split()\n        >>> sentence_chrf(ref1, hyp1) # doctest: +ELLIPSIS\n        0.6349...\n        >>> sentence_chrf(ref1, hyp2) # doctest: +ELLIPSIS\n        0.3330...\n\n    The infamous "the the the ... " example\n\n        >>> ref = \'the cat is on the mat\'.split()\n        >>> hyp = \'the the the the the the the\'.split()\n        >>> sentence_chrf(ref, hyp)  # doctest: +ELLIPSIS\n        0.1468...\n\n    An example to show that this function allows users to use strings instead of\n    tokens, i.e. list(str) as inputs.\n\n        >>> ref1 = str(\'It is a guide to action that ensures that the military \'\n        ...            \'will forever heed Party commands\')\n        >>> hyp1 = str(\'It is a guide to action which ensures that the military \'\n        ...            \'always obeys the commands of the party\')\n        >>> sentence_chrf(ref1, hyp1) # doctest: +ELLIPSIS\n        0.6349...\n        >>> type(ref1) == type(hyp1) == str\n        True\n        >>> sentence_chrf(ref1.split(), hyp1.split()) # doctest: +ELLIPSIS\n        0.6349...\n\n    To skip the unigrams and only use 2- to 3-grams:\n\n        >>> sentence_chrf(ref1, hyp1, min_len=2, max_len=3) # doctest: +ELLIPSIS\n        0.6617...\n\n    :param references: reference sentence\n    :type references: list(str) / str\n    :param hypothesis: a hypothesis sentence\n    :type hypothesis: list(str) / str\n    :param min_len: The minimum order of n-gram this function should extract.\n    :type min_len: int\n    :param max_len: The maximum order of n-gram this function should extract.\n    :type max_len: int\n    :param beta: the parameter to assign more importance to recall over precision\n    :type beta: float\n    :param ignore_whitespace: ignore whitespace characters in scoring\n    :type ignore_whitespace: bool\n    :return: the sentence level CHRF score.\n    :rtype: float\n    '
    return corpus_chrf([reference], [hypothesis], min_len, max_len, beta=beta, ignore_whitespace=ignore_whitespace)

def _preprocess(sent, ignore_whitespace):
    if False:
        for i in range(10):
            print('nop')
    if type(sent) != str:
        sent = ' '.join(sent)
    if ignore_whitespace:
        sent = re.sub('\\s+', '', sent)
    return sent

def chrf_precision_recall_fscore_support(reference, hypothesis, n, beta=3.0, epsilon=1e-16):
    if False:
        print('Hello World!')
    '\n    This function computes the precision, recall and fscore from the ngram\n    overlaps. It returns the `support` which is the true positive score.\n\n    By underspecifying the input type, the function will be agnostic as to how\n    it computes the ngrams and simply take the whichever element in the list;\n    it could be either token or character.\n\n    :param reference: The reference sentence.\n    :type reference: list\n    :param hypothesis: The hypothesis sentence.\n    :type hypothesis: list\n    :param n: Extract up to the n-th order ngrams\n    :type n: int\n    :param beta: The parameter to assign more importance to recall over precision.\n    :type beta: float\n    :param epsilon: The fallback value if the hypothesis or reference is empty.\n    :type epsilon: float\n    :return: Returns the precision, recall and f-score and support (true positive).\n    :rtype: tuple(float)\n    '
    ref_ngrams = Counter(ngrams(reference, n))
    hyp_ngrams = Counter(ngrams(hypothesis, n))
    overlap_ngrams = ref_ngrams & hyp_ngrams
    tp = sum(overlap_ngrams.values())
    tpfp = sum(hyp_ngrams.values())
    tpfn = sum(ref_ngrams.values())
    try:
        prec = tp / tpfp
        rec = tp / tpfn
        factor = beta ** 2
        fscore = (1 + factor) * (prec * rec) / (factor * prec + rec)
    except ZeroDivisionError:
        prec = rec = fscore = epsilon
    return (prec, rec, fscore, tp)

def corpus_chrf(references, hypotheses, min_len=1, max_len=6, beta=3.0, ignore_whitespace=True):
    if False:
        i = 10
        return i + 15
    "\n    Calculates the corpus level CHRF (Character n-gram F-score), it is the\n    macro-averaged value of the sentence/segment level CHRF score.\n\n    This implementation of CHRF only supports a single reference at the moment.\n\n        >>> ref1 = str('It is a guide to action that ensures that the military '\n        ...            'will forever heed Party commands').split()\n        >>> ref2 = str('It is the guiding principle which guarantees the military '\n        ...            'forces always being under the command of the Party').split()\n        >>>\n        >>> hyp1 = str('It is a guide to action which ensures that the military '\n        ...            'always obeys the commands of the party').split()\n        >>> hyp2 = str('It is to insure the troops forever hearing the activity '\n        ...            'guidebook that party direct')\n        >>> corpus_chrf([ref1, ref2, ref1, ref2], [hyp1, hyp2, hyp2, hyp1]) # doctest: +ELLIPSIS\n        0.3910...\n\n    :param references: a corpus of list of reference sentences, w.r.t. hypotheses\n    :type references: list(list(str))\n    :param hypotheses: a list of hypothesis sentences\n    :type hypotheses: list(list(str))\n    :param min_len: The minimum order of n-gram this function should extract.\n    :type min_len: int\n    :param max_len: The maximum order of n-gram this function should extract.\n    :type max_len: int\n    :param beta: the parameter to assign more importance to recall over precision\n    :type beta: float\n    :param ignore_whitespace: ignore whitespace characters in scoring\n    :type ignore_whitespace: bool\n    :return: the sentence level CHRF score.\n    :rtype: float\n    "
    assert len(references) == len(hypotheses), 'The number of hypotheses and their references should be the same'
    num_sents = len(hypotheses)
    ngram_fscores = defaultdict(lambda : list())
    for (reference, hypothesis) in zip(references, hypotheses):
        reference = _preprocess(reference, ignore_whitespace)
        hypothesis = _preprocess(hypothesis, ignore_whitespace)
        for n in range(min_len, max_len + 1):
            (prec, rec, fscore, tp) = chrf_precision_recall_fscore_support(reference, hypothesis, n, beta=beta)
            ngram_fscores[n].append(fscore)
    num_ngram_sizes = len(ngram_fscores)
    total_scores = [sum(fscores) for (n, fscores) in ngram_fscores.items()]
    return sum(total_scores) / num_ngram_sizes / num_sents