def extract(f_start, f_end, e_start, e_end, alignment, f_aligned, srctext, trgtext, srclen, trglen, max_phrase_length):
    if False:
        i = 10
        return i + 15
    '\n    This function checks for alignment point consistency and extracts\n    phrases using the chunk of consistent phrases.\n\n    A phrase pair (e, f ) is consistent with an alignment A if and only if:\n\n    (i) No English words in the phrase pair are aligned to words outside it.\n\n           ∀e i ∈ e, (e i , f j ) ∈ A ⇒ f j ∈ f\n\n    (ii) No Foreign words in the phrase pair are aligned to words outside it.\n\n            ∀f j ∈ f , (e i , f j ) ∈ A ⇒ e i ∈ e\n\n    (iii) The phrase pair contains at least one alignment point.\n\n            ∃e i ∈ e  ̄ , f j ∈ f  ̄ s.t. (e i , f j ) ∈ A\n\n    :type f_start: int\n    :param f_start: Starting index of the possible foreign language phrases\n    :type f_end: int\n    :param f_end: End index of the possible foreign language phrases\n    :type e_start: int\n    :param e_start: Starting index of the possible source language phrases\n    :type e_end: int\n    :param e_end: End index of the possible source language phrases\n    :type srctext: list\n    :param srctext: The source language tokens, a list of string.\n    :type trgtext: list\n    :param trgtext: The target language tokens, a list of string.\n    :type srclen: int\n    :param srclen: The number of tokens in the source language tokens.\n    :type trglen: int\n    :param trglen: The number of tokens in the target language tokens.\n    '
    if f_end < 0:
        return {}
    for (e, f) in alignment:
        if f_start <= f <= f_end and (e < e_start or e > e_end):
            return {}
    phrases = set()
    fs = f_start
    while True:
        fe = min(f_end, f_start + max_phrase_length - 1)
        while True:
            src_phrase = ' '.join(srctext[e_start:e_end + 1])
            trg_phrase = ' '.join(trgtext[fs:fe + 1])
            phrases.add(((e_start, e_end + 1), (fs, fe + 1), src_phrase, trg_phrase))
            fe += 1
            if fe in f_aligned or fe >= trglen:
                break
        fs -= 1
        if fs in f_aligned or fs < 0:
            break
    return phrases

def phrase_extraction(srctext, trgtext, alignment, max_phrase_length=0):
    if False:
        i = 10
        return i + 15
    '\n    Phrase extraction algorithm extracts all consistent phrase pairs from\n    a word-aligned sentence pair.\n\n    The idea is to loop over all possible source language (e) phrases and find\n    the minimal foreign phrase (f) that matches each of them. Matching is done\n    by identifying all alignment points for the source phrase and finding the\n    shortest foreign phrase that includes all the foreign counterparts for the\n    source words.\n\n    In short, a phrase alignment has to\n    (a) contain all alignment points for all covered words\n    (b) contain at least one alignment point\n\n    >>> srctext = "michael assumes that he will stay in the house"\n    >>> trgtext = "michael geht davon aus , dass er im haus bleibt"\n    >>> alignment = [(0,0), (1,1), (1,2), (1,3), (2,5), (3,6), (4,9),\n    ... (5,9), (6,7), (7,7), (8,8)]\n    >>> phrases = phrase_extraction(srctext, trgtext, alignment)\n    >>> for i in sorted(phrases):\n    ...    print(i)\n    ...\n    ((0, 1), (0, 1), \'michael\', \'michael\')\n    ((0, 2), (0, 4), \'michael assumes\', \'michael geht davon aus\')\n    ((0, 2), (0, 5), \'michael assumes\', \'michael geht davon aus ,\')\n    ((0, 3), (0, 6), \'michael assumes that\', \'michael geht davon aus , dass\')\n    ((0, 4), (0, 7), \'michael assumes that he\', \'michael geht davon aus , dass er\')\n    ((0, 9), (0, 10), \'michael assumes that he will stay in the house\', \'michael geht davon aus , dass er im haus bleibt\')\n    ((1, 2), (1, 4), \'assumes\', \'geht davon aus\')\n    ((1, 2), (1, 5), \'assumes\', \'geht davon aus ,\')\n    ((1, 3), (1, 6), \'assumes that\', \'geht davon aus , dass\')\n    ((1, 4), (1, 7), \'assumes that he\', \'geht davon aus , dass er\')\n    ((1, 9), (1, 10), \'assumes that he will stay in the house\', \'geht davon aus , dass er im haus bleibt\')\n    ((2, 3), (4, 6), \'that\', \', dass\')\n    ((2, 3), (5, 6), \'that\', \'dass\')\n    ((2, 4), (4, 7), \'that he\', \', dass er\')\n    ((2, 4), (5, 7), \'that he\', \'dass er\')\n    ((2, 9), (4, 10), \'that he will stay in the house\', \', dass er im haus bleibt\')\n    ((2, 9), (5, 10), \'that he will stay in the house\', \'dass er im haus bleibt\')\n    ((3, 4), (6, 7), \'he\', \'er\')\n    ((3, 9), (6, 10), \'he will stay in the house\', \'er im haus bleibt\')\n    ((4, 6), (9, 10), \'will stay\', \'bleibt\')\n    ((4, 9), (7, 10), \'will stay in the house\', \'im haus bleibt\')\n    ((6, 8), (7, 8), \'in the\', \'im\')\n    ((6, 9), (7, 9), \'in the house\', \'im haus\')\n    ((8, 9), (8, 9), \'house\', \'haus\')\n\n    :type srctext: str\n    :param srctext: The sentence string from the source language.\n    :type trgtext: str\n    :param trgtext: The sentence string from the target language.\n    :type alignment: list(tuple)\n    :param alignment: The word alignment outputs as list of tuples, where\n        the first elements of tuples are the source words\' indices and\n        second elements are the target words\' indices. This is also the output\n        format of nltk.translate.ibm1\n    :rtype: list(tuple)\n    :return: A list of tuples, each element in a list is a phrase and each\n        phrase is a tuple made up of (i) its source location, (ii) its target\n        location, (iii) the source phrase and (iii) the target phrase. The phrase\n        list of tuples represents all the possible phrases extracted from the\n        word alignments.\n    :type max_phrase_length: int\n    :param max_phrase_length: maximal phrase length, if 0 or not specified\n        it is set to a length of the longer sentence (srctext or trgtext).\n    '
    srctext = srctext.split()
    trgtext = trgtext.split()
    srclen = len(srctext)
    trglen = len(trgtext)
    f_aligned = [j for (_, j) in alignment]
    max_phrase_length = max_phrase_length or max(srclen, trglen)
    bp = set()
    for e_start in range(srclen):
        max_idx = min(srclen, e_start + max_phrase_length)
        for e_end in range(e_start, max_idx):
            (f_start, f_end) = (trglen - 1, -1)
            for (e, f) in alignment:
                if e_start <= e <= e_end:
                    f_start = min(f, f_start)
                    f_end = max(f, f_end)
            phrases = extract(f_start, f_end, e_start, e_end, alignment, f_aligned, srctext, trgtext, srclen, trglen, max_phrase_length)
            if phrases:
                bp.update(phrases)
    return bp