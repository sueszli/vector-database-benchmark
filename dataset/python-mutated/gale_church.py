"""

A port of the Gale-Church Aligner.

Gale & Church (1993), A Program for Aligning Sentences in Bilingual Corpora.
https://aclweb.org/anthology/J93-1004.pdf

"""
import math
try:
    from norm import logsf as norm_logsf
    from scipy.stats import norm
except ImportError:

    def erfcc(x):
        if False:
            print('Hello World!')
        'Complementary error function.'
        z = abs(x)
        t = 1 / (1 + 0.5 * z)
        r = t * math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
        if x >= 0.0:
            return r
        else:
            return 2.0 - r

    def norm_cdf(x):
        if False:
            return 10
        'Return the area under the normal distribution from M{-∞..x}.'
        return 1 - 0.5 * erfcc(x / math.sqrt(2))

    def norm_logsf(x):
        if False:
            while True:
                i = 10
        try:
            return math.log(1 - norm_cdf(x))
        except ValueError:
            return float('-inf')
LOG2 = math.log(2)

class LanguageIndependent:
    PRIORS = {(1, 0): 0.0099, (0, 1): 0.0099, (1, 1): 0.89, (2, 1): 0.089, (1, 2): 0.089, (2, 2): 0.011}
    AVERAGE_CHARACTERS = 1
    VARIANCE_CHARACTERS = 6.8

def trace(backlinks, source_sents_lens, target_sents_lens):
    if False:
        while True:
            i = 10
    "\n    Traverse the alignment cost from the tracebacks and retrieves\n    appropriate sentence pairs.\n\n    :param backlinks: A dictionary where the key is the alignment points and value is the cost (referencing the LanguageIndependent.PRIORS)\n    :type backlinks: dict\n    :param source_sents_lens: A list of target sentences' lengths\n    :type source_sents_lens: list(int)\n    :param target_sents_lens: A list of target sentences' lengths\n    :type target_sents_lens: list(int)\n    "
    links = []
    position = (len(source_sents_lens), len(target_sents_lens))
    while position != (0, 0) and all((p >= 0 for p in position)):
        try:
            (s, t) = backlinks[position]
        except TypeError:
            position = (position[0] - 1, position[1] - 1)
            continue
        for i in range(s):
            for j in range(t):
                links.append((position[0] - i - 1, position[1] - j - 1))
        position = (position[0] - s, position[1] - t)
    return links[::-1]

def align_log_prob(i, j, source_sents, target_sents, alignment, params):
    if False:
        i = 10
        return i + 15
    'Returns the log probability of the two sentences C{source_sents[i]}, C{target_sents[j]}\n    being aligned with a specific C{alignment}.\n\n    @param i: The offset of the source sentence.\n    @param j: The offset of the target sentence.\n    @param source_sents: The list of source sentence lengths.\n    @param target_sents: The list of target sentence lengths.\n    @param alignment: The alignment type, a tuple of two integers.\n    @param params: The sentence alignment parameters.\n\n    @returns: The log probability of a specific alignment between the two sentences, given the parameters.\n    '
    l_s = sum((source_sents[i - offset - 1] for offset in range(alignment[0])))
    l_t = sum((target_sents[j - offset - 1] for offset in range(alignment[1])))
    try:
        m = (l_s + l_t / params.AVERAGE_CHARACTERS) / 2
        delta = (l_s * params.AVERAGE_CHARACTERS - l_t) / math.sqrt(m * params.VARIANCE_CHARACTERS)
    except ZeroDivisionError:
        return float('-inf')
    return -(LOG2 + norm_logsf(abs(delta)) + math.log(params.PRIORS[alignment]))

def align_blocks(source_sents_lens, target_sents_lens, params=LanguageIndependent):
    if False:
        for i in range(10):
            print('nop')
    'Return the sentence alignment of two text blocks (usually paragraphs).\n\n        >>> align_blocks([5,5,5], [7,7,7])\n        [(0, 0), (1, 1), (2, 2)]\n        >>> align_blocks([10,5,5], [12,20])\n        [(0, 0), (1, 1), (2, 1)]\n        >>> align_blocks([12,20], [10,5,5])\n        [(0, 0), (1, 1), (1, 2)]\n        >>> align_blocks([10,2,10,10,2,10], [12,3,20,3,12])\n        [(0, 0), (1, 1), (2, 2), (3, 2), (4, 3), (5, 4)]\n\n    @param source_sents_lens: The list of source sentence lengths.\n    @param target_sents_lens: The list of target sentence lengths.\n    @param params: the sentence alignment parameters.\n    @return: The sentence alignments, a list of index pairs.\n    '
    alignment_types = list(params.PRIORS.keys())
    D = [[]]
    backlinks = {}
    for i in range(len(source_sents_lens) + 1):
        for j in range(len(target_sents_lens) + 1):
            min_dist = float('inf')
            min_align = None
            for a in alignment_types:
                prev_i = -1 - a[0]
                prev_j = j - a[1]
                if prev_i < -len(D) or prev_j < 0:
                    continue
                p = D[prev_i][prev_j] + align_log_prob(i, j, source_sents_lens, target_sents_lens, a, params)
                if p < min_dist:
                    min_dist = p
                    min_align = a
            if min_dist == float('inf'):
                min_dist = 0
            backlinks[i, j] = min_align
            D[-1].append(min_dist)
        if len(D) > 2:
            D.pop(0)
        D.append([])
    return trace(backlinks, source_sents_lens, target_sents_lens)

def align_texts(source_blocks, target_blocks, params=LanguageIndependent):
    if False:
        i = 10
        return i + 15
    'Creates the sentence alignment of two texts.\n\n    Texts can consist of several blocks. Block boundaries cannot be crossed by sentence\n    alignment links.\n\n    Each block consists of a list that contains the lengths (in characters) of the sentences\n    in this block.\n\n    @param source_blocks: The list of blocks in the source text.\n    @param target_blocks: The list of blocks in the target text.\n    @param params: the sentence alignment parameters.\n\n    @returns: A list of sentence alignment lists\n    '
    if len(source_blocks) != len(target_blocks):
        raise ValueError('Source and target texts do not have the same number of blocks.')
    return [align_blocks(source_block, target_block, params) for (source_block, target_block) in zip(source_blocks, target_blocks)]

def split_at(it, split_value):
    if False:
        for i in range(10):
            print('nop')
    'Splits an iterator C{it} at values of C{split_value}.\n\n    Each instance of C{split_value} is swallowed. The iterator produces\n    subiterators which need to be consumed fully before the next subiterator\n    can be used.\n    '

    def _chunk_iterator(first):
        if False:
            for i in range(10):
                print('nop')
        v = first
        while v != split_value:
            yield v
            v = it.next()
    while True:
        yield _chunk_iterator(it.next())

def parse_token_stream(stream, soft_delimiter, hard_delimiter):
    if False:
        i = 10
        return i + 15
    'Parses a stream of tokens and splits it into sentences (using C{soft_delimiter} tokens)\n    and blocks (using C{hard_delimiter} tokens) for use with the L{align_texts} function.\n    '
    return [[sum((len(token) for token in sentence_it)) for sentence_it in split_at(block_it, soft_delimiter)] for block_it in split_at(stream, hard_delimiter)]