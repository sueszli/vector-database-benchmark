"""This module contains functions to compute direct confirmation on a pair of words or word subsets."""
import logging
import numpy as np
logger = logging.getLogger(__name__)
EPSILON = 1e-12

def log_conditional_probability(segmented_topics, accumulator, with_std=False, with_support=False):
    if False:
        for i in range(10):
            print('nop')
    "Calculate the log-conditional-probability measure which is used by coherence measures such as `U_mass`.\n    This is defined as :math:`m_{lc}(S_i) = log \\frac{P(W', W^{*}) + \\epsilon}{P(W^{*})}`.\n\n    Parameters\n    ----------\n    segmented_topics : list of lists of (int, int)\n        Output from the :func:`~gensim.topic_coherence.segmentation.s_one_pre`,\n        :func:`~gensim.topic_coherence.segmentation.s_one_one`.\n    accumulator : :class:`~gensim.topic_coherence.text_analysis.InvertedIndexAccumulator`\n        Word occurrence accumulator from :mod:`gensim.topic_coherence.probability_estimation`.\n    with_std : bool, optional\n        True to also include standard deviation across topic segment sets in addition to the mean coherence\n        for each topic.\n    with_support : bool, optional\n        True to also include support across topic segments. The support is defined as the number of pairwise\n        similarity comparisons were used to compute the overall topic coherence.\n\n    Returns\n    -------\n    list of float\n        Log conditional probabilities measurement for each topic.\n\n    Examples\n    --------\n    .. sourcecode:: pycon\n\n        >>> from gensim.topic_coherence import direct_confirmation_measure, text_analysis\n        >>> from collections import namedtuple\n        >>>\n        >>> # Create dictionary\n        >>> id2token = {1: 'test', 2: 'doc'}\n        >>> token2id = {v: k for k, v in id2token.items()}\n        >>> dictionary = namedtuple('Dictionary', 'token2id, id2token')(token2id, id2token)\n        >>>\n        >>> # Initialize segmented topics and accumulator\n        >>> segmentation = [[(1, 2)]]\n        >>>\n        >>> accumulator = text_analysis.InvertedIndexAccumulator({1, 2}, dictionary)\n        >>> accumulator._inverted_index = {0: {2, 3, 4}, 1: {3, 5}}\n        >>> accumulator._num_docs = 5\n        >>>\n        >>> # result should be ~ ln(1 / 2) = -0.693147181\n        >>> result = direct_confirmation_measure.log_conditional_probability(segmentation, accumulator)[0]\n\n    "
    topic_coherences = []
    num_docs = float(accumulator.num_docs)
    for s_i in segmented_topics:
        segment_sims = []
        for (w_prime, w_star) in s_i:
            try:
                w_star_count = accumulator[w_star]
                co_occur_count = accumulator[w_prime, w_star]
                m_lc_i = np.log((co_occur_count / num_docs + EPSILON) / (w_star_count / num_docs))
            except KeyError:
                m_lc_i = 0.0
            except ZeroDivisionError:
                m_lc_i = 0.0
            segment_sims.append(m_lc_i)
        topic_coherences.append(aggregate_segment_sims(segment_sims, with_std, with_support))
    return topic_coherences

def aggregate_segment_sims(segment_sims, with_std, with_support):
    if False:
        print('Hello World!')
    'Compute various statistics from the segment similarities generated via set pairwise comparisons\n    of top-N word lists for a single topic.\n\n    Parameters\n    ----------\n    segment_sims : iterable of float\n        Similarity values to aggregate.\n    with_std : bool\n        Set to True to include standard deviation.\n    with_support : bool\n        Set to True to include number of elements in `segment_sims` as a statistic in the results returned.\n\n    Returns\n    -------\n    (float[, float[, int]])\n        Tuple with (mean[, std[, support]]).\n\n    Examples\n    ---------\n    .. sourcecode:: pycon\n\n        >>> from gensim.topic_coherence import direct_confirmation_measure\n        >>>\n        >>> segment_sims = [0.2, 0.5, 1., 0.05]\n        >>> direct_confirmation_measure.aggregate_segment_sims(segment_sims, True, True)\n        (0.4375, 0.36293077852394939, 4)\n        >>> direct_confirmation_measure.aggregate_segment_sims(segment_sims, False, False)\n        0.4375\n\n    '
    mean = np.mean(segment_sims)
    stats = [mean]
    if with_std:
        stats.append(np.std(segment_sims))
    if with_support:
        stats.append(len(segment_sims))
    return stats[0] if len(stats) == 1 else tuple(stats)

def log_ratio_measure(segmented_topics, accumulator, normalize=False, with_std=False, with_support=False):
    if False:
        for i in range(10):
            print('nop')
    'Compute log ratio measure for `segment_topics`.\n\n    Parameters\n    ----------\n    segmented_topics : list of lists of (int, int)\n        Output from the :func:`~gensim.topic_coherence.segmentation.s_one_pre`,\n        :func:`~gensim.topic_coherence.segmentation.s_one_one`.\n    accumulator : :class:`~gensim.topic_coherence.text_analysis.InvertedIndexAccumulator`\n        Word occurrence accumulator from :mod:`gensim.topic_coherence.probability_estimation`.\n    normalize : bool, optional\n        Details in the "Notes" section.\n    with_std : bool, optional\n        True to also include standard deviation across topic segment sets in addition to the mean coherence\n        for each topic.\n    with_support : bool, optional\n        True to also include support across topic segments. The support is defined as the number of pairwise\n        similarity comparisons were used to compute the overall topic coherence.\n\n    Notes\n    -----\n    If `normalize=False`:\n        Calculate the log-ratio-measure, popularly known as **PMI** which is used by coherence measures such as `c_v`.\n        This is defined as :math:`m_{lr}(S_i) = log \\frac{P(W\', W^{*}) + \\epsilon}{P(W\') * P(W^{*})}`\n\n    If `normalize=True`:\n        Calculate the normalized-log-ratio-measure, popularly knowns as **NPMI**\n        which is used by coherence measures such as `c_v`.\n        This is defined as :math:`m_{nlr}(S_i) = \\frac{m_{lr}(S_i)}{-log(P(W\', W^{*}) + \\epsilon)}`\n\n    Returns\n    -------\n    list of float\n        Log ratio measurements for each topic.\n\n    Examples\n    --------\n    .. sourcecode:: pycon\n\n        >>> from gensim.topic_coherence import direct_confirmation_measure, text_analysis\n        >>> from collections import namedtuple\n        >>>\n        >>> # Create dictionary\n        >>> id2token = {1: \'test\', 2: \'doc\'}\n        >>> token2id = {v: k for k, v in id2token.items()}\n        >>> dictionary = namedtuple(\'Dictionary\', \'token2id, id2token\')(token2id, id2token)\n        >>>\n        >>> # Initialize segmented topics and accumulator\n        >>> segmentation = [[(1, 2)]]\n        >>>\n        >>> accumulator = text_analysis.InvertedIndexAccumulator({1, 2}, dictionary)\n        >>> accumulator._inverted_index = {0: {2, 3, 4}, 1: {3, 5}}\n        >>> accumulator._num_docs = 5\n        >>>\n        >>> # result should be ~ ln{(1 / 5) / [(3 / 5) * (2 / 5)]} = -0.182321557\n        >>> result = direct_confirmation_measure.log_ratio_measure(segmentation, accumulator)[0]\n\n    '
    topic_coherences = []
    num_docs = float(accumulator.num_docs)
    for s_i in segmented_topics:
        segment_sims = []
        for (w_prime, w_star) in s_i:
            w_prime_count = accumulator[w_prime]
            w_star_count = accumulator[w_star]
            co_occur_count = accumulator[w_prime, w_star]
            if normalize:
                numerator = log_ratio_measure([[(w_prime, w_star)]], accumulator)[0]
                co_doc_prob = co_occur_count / num_docs
                m_lr_i = numerator / -np.log(co_doc_prob + EPSILON)
            else:
                numerator = co_occur_count / num_docs + EPSILON
                denominator = w_prime_count / num_docs * (w_star_count / num_docs)
                m_lr_i = np.log(numerator / denominator)
            segment_sims.append(m_lr_i)
        topic_coherences.append(aggregate_segment_sims(segment_sims, with_std, with_support))
    return topic_coherences