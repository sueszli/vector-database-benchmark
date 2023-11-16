""" RIBES score implementation """
import math
from itertools import islice
from nltk.util import choose, ngrams

def sentence_ribes(references, hypothesis, alpha=0.25, beta=0.1):
    if False:
        i = 10
        return i + 15
    '\n    The RIBES (Rank-based Intuitive Bilingual Evaluation Score) from\n    Hideki Isozaki, Tsutomu Hirao, Kevin Duh, Katsuhito Sudoh and\n    Hajime Tsukada. 2010. "Automatic Evaluation of Translation Quality for\n    Distant Language Pairs". In Proceedings of EMNLP.\n    https://www.aclweb.org/anthology/D/D10/D10-1092.pdf\n\n    The generic RIBES scores used in shared task, e.g. Workshop for\n    Asian Translation (WAT) uses the following RIBES calculations:\n\n        RIBES = kendall_tau * (alpha**p1) * (beta**bp)\n\n    Please note that this re-implementation differs from the official\n    RIBES implementation and though it emulates the results as describe\n    in the original paper, there are further optimization implemented\n    in the official RIBES script.\n\n    Users are encouraged to use the official RIBES script instead of this\n    implementation when evaluating your machine translation system. Refer\n    to https://www.kecl.ntt.co.jp/icl/lirg/ribes/ for the official script.\n\n    :param references: a list of reference sentences\n    :type references: list(list(str))\n    :param hypothesis: a hypothesis sentence\n    :type hypothesis: list(str)\n    :param alpha: hyperparameter used as a prior for the unigram precision.\n    :type alpha: float\n    :param beta: hyperparameter used as a prior for the brevity penalty.\n    :type beta: float\n    :return: The best ribes score from one of the references.\n    :rtype: float\n    '
    best_ribes = -1.0
    for reference in references:
        worder = word_rank_alignment(reference, hypothesis)
        nkt = kendall_tau(worder)
        bp = min(1.0, math.exp(1.0 - len(reference) / len(hypothesis)))
        p1 = len(worder) / len(hypothesis)
        _ribes = nkt * p1 ** alpha * bp ** beta
        if _ribes > best_ribes:
            best_ribes = _ribes
    return best_ribes

def corpus_ribes(list_of_references, hypotheses, alpha=0.25, beta=0.1):
    if False:
        for i in range(10):
            print('nop')
    '\n    This function "calculates RIBES for a system output (hypothesis) with\n    multiple references, and returns "best" score among multi-references and\n    individual scores. The scores are corpus-wise, i.e., averaged by the number\n    of sentences." (c.f. RIBES version 1.03.1 code).\n\n    Different from BLEU\'s micro-average precision, RIBES calculates the\n    macro-average precision by averaging the best RIBES score for each pair of\n    hypothesis and its corresponding references\n\n    >>> hyp1 = [\'It\', \'is\', \'a\', \'guide\', \'to\', \'action\', \'which\',\n    ...         \'ensures\', \'that\', \'the\', \'military\', \'always\',\n    ...         \'obeys\', \'the\', \'commands\', \'of\', \'the\', \'party\']\n    >>> ref1a = [\'It\', \'is\', \'a\', \'guide\', \'to\', \'action\', \'that\',\n    ...          \'ensures\', \'that\', \'the\', \'military\', \'will\', \'forever\',\n    ...          \'heed\', \'Party\', \'commands\']\n    >>> ref1b = [\'It\', \'is\', \'the\', \'guiding\', \'principle\', \'which\',\n    ...          \'guarantees\', \'the\', \'military\', \'forces\', \'always\',\n    ...          \'being\', \'under\', \'the\', \'command\', \'of\', \'the\', \'Party\']\n    >>> ref1c = [\'It\', \'is\', \'the\', \'practical\', \'guide\', \'for\', \'the\',\n    ...          \'army\', \'always\', \'to\', \'heed\', \'the\', \'directions\',\n    ...          \'of\', \'the\', \'party\']\n\n    >>> hyp2 = [\'he\', \'read\', \'the\', \'book\', \'because\', \'he\', \'was\',\n    ...         \'interested\', \'in\', \'world\', \'history\']\n    >>> ref2a = [\'he\', \'was\', \'interested\', \'in\', \'world\', \'history\',\n    ...          \'because\', \'he\', \'read\', \'the\', \'book\']\n\n    >>> list_of_references = [[ref1a, ref1b, ref1c], [ref2a]]\n    >>> hypotheses = [hyp1, hyp2]\n    >>> round(corpus_ribes(list_of_references, hypotheses),4)\n    0.3597\n\n    :param references: a corpus of lists of reference sentences, w.r.t. hypotheses\n    :type references: list(list(list(str)))\n    :param hypotheses: a list of hypothesis sentences\n    :type hypotheses: list(list(str))\n    :param alpha: hyperparameter used as a prior for the unigram precision.\n    :type alpha: float\n    :param beta: hyperparameter used as a prior for the brevity penalty.\n    :type beta: float\n    :return: The best ribes score from one of the references.\n    :rtype: float\n    '
    corpus_best_ribes = 0.0
    for (references, hypothesis) in zip(list_of_references, hypotheses):
        corpus_best_ribes += sentence_ribes(references, hypothesis, alpha, beta)
    return corpus_best_ribes / len(hypotheses)

def position_of_ngram(ngram, sentence):
    if False:
        for i in range(10):
            print('nop')
    "\n    This function returns the position of the first instance of the ngram\n    appearing in a sentence.\n\n    Note that one could also use string as follows but the code is a little\n    convoluted with type casting back and forth:\n\n        char_pos = ' '.join(sent)[:' '.join(sent).index(' '.join(ngram))]\n        word_pos = char_pos.count(' ')\n\n    Another way to conceive this is:\n\n        return next(i for i, ng in enumerate(ngrams(sentence, len(ngram)))\n                    if ng == ngram)\n\n    :param ngram: The ngram that needs to be searched\n    :type ngram: tuple\n    :param sentence: The list of tokens to search from.\n    :type sentence: list(str)\n    "
    for (i, sublist) in enumerate(ngrams(sentence, len(ngram))):
        if ngram == sublist:
            return i

def word_rank_alignment(reference, hypothesis, character_based=False):
    if False:
        while True:
            i = 10
    "\n    This is the word rank alignment algorithm described in the paper to produce\n    the *worder* list, i.e. a list of word indices of the hypothesis word orders\n    w.r.t. the list of reference words.\n\n    Below is (H0, R0) example from the Isozaki et al. 2010 paper,\n    note the examples are indexed from 1 but the results here are indexed from 0:\n\n        >>> ref = str('he was interested in world history because he '\n        ... 'read the book').split()\n        >>> hyp = str('he read the book because he was interested in world '\n        ... 'history').split()\n        >>> word_rank_alignment(ref, hyp)\n        [7, 8, 9, 10, 6, 0, 1, 2, 3, 4, 5]\n\n    The (H1, R1) example from the paper, note the 0th index:\n\n        >>> ref = 'John hit Bob yesterday'.split()\n        >>> hyp = 'Bob hit John yesterday'.split()\n        >>> word_rank_alignment(ref, hyp)\n        [2, 1, 0, 3]\n\n    Here is the (H2, R2) example from the paper, note the 0th index here too:\n\n        >>> ref = 'the boy read the book'.split()\n        >>> hyp = 'the book was read by the boy'.split()\n        >>> word_rank_alignment(ref, hyp)\n        [3, 4, 2, 0, 1]\n\n    :param reference: a reference sentence\n    :type reference: list(str)\n    :param hypothesis: a hypothesis sentence\n    :type hypothesis: list(str)\n    "
    worder = []
    hyp_len = len(hypothesis)
    ref_ngrams = []
    hyp_ngrams = []
    for n in range(1, len(reference) + 1):
        for ng in ngrams(reference, n):
            ref_ngrams.append(ng)
        for ng in ngrams(hypothesis, n):
            hyp_ngrams.append(ng)
    for (i, h_word) in enumerate(hypothesis):
        if h_word not in reference:
            continue
        elif hypothesis.count(h_word) == reference.count(h_word) == 1:
            worder.append(reference.index(h_word))
        else:
            max_window_size = max(i, hyp_len - i + 1)
            for window in range(1, max_window_size):
                if i + window < hyp_len:
                    right_context_ngram = tuple(islice(hypothesis, i, i + window + 1))
                    num_times_in_ref = ref_ngrams.count(right_context_ngram)
                    num_times_in_hyp = hyp_ngrams.count(right_context_ngram)
                    if num_times_in_ref == num_times_in_hyp == 1:
                        pos = position_of_ngram(right_context_ngram, reference)
                        worder.append(pos)
                        break
                if window <= i:
                    left_context_ngram = tuple(islice(hypothesis, i - window, i + 1))
                    num_times_in_ref = ref_ngrams.count(left_context_ngram)
                    num_times_in_hyp = hyp_ngrams.count(left_context_ngram)
                    if num_times_in_ref == num_times_in_hyp == 1:
                        pos = position_of_ngram(left_context_ngram, reference)
                        worder.append(pos + len(left_context_ngram) - 1)
                        break
    return worder

def find_increasing_sequences(worder):
    if False:
        return 10
    '\n    Given the *worder* list, this function groups monotonic +1 sequences.\n\n        >>> worder = [7, 8, 9, 10, 6, 0, 1, 2, 3, 4, 5]\n        >>> list(find_increasing_sequences(worder))\n        [(7, 8, 9, 10), (0, 1, 2, 3, 4, 5)]\n\n    :param worder: The worder list output from word_rank_alignment\n    :param type: list(int)\n    '
    items = iter(worder)
    (a, b) = (None, next(items, None))
    result = [b]
    while b is not None:
        (a, b) = (b, next(items, None))
        if b is not None and a + 1 == b:
            result.append(b)
        else:
            if len(result) > 1:
                yield tuple(result)
            result = [b]

def kendall_tau(worder, normalize=True):
    if False:
        i = 10
        return i + 15
    "\n    Calculates the Kendall's Tau correlation coefficient given the *worder*\n    list of word alignments from word_rank_alignment(), using the formula:\n\n        tau = 2 * num_increasing_pairs / num_possible_pairs -1\n\n    Note that the no. of increasing pairs can be discontinuous in the *worder*\n    list and each each increasing sequence can be tabulated as choose(len(seq), 2)\n    no. of increasing pairs, e.g.\n\n        >>> worder = [7, 8, 9, 10, 6, 0, 1, 2, 3, 4, 5]\n        >>> number_possible_pairs = choose(len(worder), 2)\n        >>> round(kendall_tau(worder, normalize=False),3)\n        -0.236\n        >>> round(kendall_tau(worder),3)\n        0.382\n\n    :param worder: The worder list output from word_rank_alignment\n    :type worder: list(int)\n    :param normalize: Flag to indicate normalization to between 0.0 and 1.0.\n    :type normalize: boolean\n    :return: The Kendall's Tau correlation coefficient.\n    :rtype: float\n    "
    worder_len = len(worder)
    if worder_len < 2:
        tau = -1
    else:
        increasing_sequences = find_increasing_sequences(worder)
        num_increasing_pairs = sum((choose(len(seq), 2) for seq in increasing_sequences))
        num_possible_pairs = choose(worder_len, 2)
        tau = 2 * num_increasing_pairs / num_possible_pairs - 1
    if normalize:
        return (tau + 1) / 2
    else:
        return tau

def spearman_rho(worder, normalize=True):
    if False:
        i = 10
        return i + 15
    "\n    Calculates the Spearman's Rho correlation coefficient given the *worder*\n    list of word alignment from word_rank_alignment(), using the formula:\n\n        rho = 1 - sum(d**2) / choose(len(worder)+1, 3)\n\n    Given that d is the sum of difference between the *worder* list of indices\n    and the original word indices from the reference sentence.\n\n    Using the (H0,R0) and (H5, R5) example from the paper\n\n        >>> worder =  [7, 8, 9, 10, 6, 0, 1, 2, 3, 4, 5]\n        >>> round(spearman_rho(worder, normalize=False), 3)\n        -0.591\n        >>> round(spearman_rho(worder), 3)\n        0.205\n\n    :param worder: The worder list output from word_rank_alignment\n    :param type: list(int)\n    "
    worder_len = len(worder)
    sum_d_square = sum(((wi - i) ** 2 for (wi, i) in zip(worder, range(worder_len))))
    rho = 1 - sum_d_square / choose(worder_len + 1, 3)
    if normalize:
        return (rho + 1) / 2
    else:
        return rho