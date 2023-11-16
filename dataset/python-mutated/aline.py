"""
ALINE
https://webdocs.cs.ualberta.ca/~kondrak/
Copyright 2002 by Grzegorz Kondrak.

ALINE is an algorithm for aligning phonetic sequences, described in [1].
This module is a port of Kondrak's (2002) ALINE. It provides functions for
phonetic sequence alignment and similarity analysis. These are useful in
historical linguistics, sociolinguistics and synchronic phonology.

ALINE has parameters that can be tuned for desired output. These parameters are:
- C_skip, C_sub, C_exp, C_vwl
- Salience weights
- Segmental features

In this implementation, some parameters have been changed from their default
values as described in [1], in order to replicate published results. All changes
are noted in comments.

Example usage
-------------

# Get optimal alignment of two phonetic sequences

>>> align('θin', 'tenwis') # doctest: +SKIP
[[('θ', 't'), ('i', 'e'), ('n', 'n'), ('-', 'w'), ('-', 'i'), ('-', 's')]]

[1] G. Kondrak. Algorithms for Language Reconstruction. PhD dissertation,
University of Toronto.
"""
try:
    import numpy as np
except ImportError:
    np = None
inf = float('inf')
C_skip = -10
C_sub = 35
C_exp = 45
C_vwl = 5
consonants = ['B', 'N', 'R', 'b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'x', 'z', 'ç', 'ð', 'ħ', 'ŋ', 'ɖ', 'ɟ', 'ɢ', 'ɣ', 'ɦ', 'ɬ', 'ɮ', 'ɰ', 'ɱ', 'ɲ', 'ɳ', 'ɴ', 'ɸ', 'ɹ', 'ɻ', 'ɽ', 'ɾ', 'ʀ', 'ʁ', 'ʂ', 'ʃ', 'ʈ', 'ʋ', 'ʐ ', 'ʒ', 'ʔ', 'ʕ', 'ʙ', 'ʝ', 'β', 'θ', 'χ', 'ʐ', 'w']
R_c = ['aspirated', 'lateral', 'manner', 'nasal', 'place', 'retroflex', 'syllabic', 'voice']
R_v = ['back', 'lateral', 'long', 'manner', 'nasal', 'place', 'retroflex', 'round', 'syllabic', 'voice']
similarity_matrix = {'bilabial': 1.0, 'labiodental': 0.95, 'dental': 0.9, 'alveolar': 0.85, 'retroflex': 0.8, 'palato-alveolar': 0.75, 'palatal': 0.7, 'velar': 0.6, 'uvular': 0.5, 'pharyngeal': 0.3, 'glottal': 0.1, 'labiovelar': 1.0, 'vowel': -1.0, 'stop': 1.0, 'affricate': 0.9, 'fricative': 0.85, 'trill': 0.7, 'tap': 0.65, 'approximant': 0.6, 'high vowel': 0.4, 'mid vowel': 0.2, 'low vowel': 0.0, 'vowel2': 0.5, 'high': 1.0, 'mid': 0.5, 'low': 0.0, 'front': 1.0, 'central': 0.5, 'back': 0.0, 'plus': 1.0, 'minus': 0.0}
salience = {'syllabic': 5, 'place': 40, 'manner': 50, 'voice': 5, 'nasal': 20, 'retroflex': 10, 'lateral': 10, 'aspirated': 5, 'long': 0, 'high': 3, 'back': 2, 'round': 2}
feature_matrix = {'p': {'place': 'bilabial', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'minus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'b': {'place': 'bilabial', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 't': {'place': 'alveolar', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'minus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'd': {'place': 'alveolar', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ʈ': {'place': 'retroflex', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'minus', 'nasal': 'minus', 'retroflex': 'plus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ɖ': {'place': 'retroflex', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'plus', 'lateral': 'minus', 'aspirated': 'minus'}, 'c': {'place': 'palatal', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'minus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ɟ': {'place': 'palatal', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'k': {'place': 'velar', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'minus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'g': {'place': 'velar', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'q': {'place': 'uvular', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'minus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ɢ': {'place': 'uvular', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ʔ': {'place': 'glottal', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'minus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'm': {'place': 'bilabial', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'plus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ɱ': {'place': 'labiodental', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'plus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'n': {'place': 'alveolar', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'plus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ɳ': {'place': 'retroflex', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'plus', 'retroflex': 'plus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ɲ': {'place': 'palatal', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'plus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ŋ': {'place': 'velar', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'plus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ɴ': {'place': 'uvular', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'plus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'N': {'place': 'uvular', 'manner': 'stop', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'plus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ʙ': {'place': 'bilabial', 'manner': 'trill', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'B': {'place': 'bilabial', 'manner': 'trill', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'r': {'place': 'alveolar', 'manner': 'trill', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'plus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ʀ': {'place': 'uvular', 'manner': 'trill', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'R': {'place': 'uvular', 'manner': 'trill', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ɾ': {'place': 'alveolar', 'manner': 'tap', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ɽ': {'place': 'retroflex', 'manner': 'tap', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'plus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ɸ': {'place': 'bilabial', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'minus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'β': {'place': 'bilabial', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'f': {'place': 'labiodental', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'minus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'v': {'place': 'labiodental', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'θ': {'place': 'dental', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'minus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ð': {'place': 'dental', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 's': {'place': 'alveolar', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'minus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'z': {'place': 'alveolar', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ʃ': {'place': 'palato-alveolar', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'minus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ʒ': {'place': 'palato-alveolar', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ʂ': {'place': 'retroflex', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'minus', 'nasal': 'minus', 'retroflex': 'plus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ʐ': {'place': 'retroflex', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'plus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ç': {'place': 'palatal', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'minus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ʝ': {'place': 'palatal', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'x': {'place': 'velar', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'minus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ɣ': {'place': 'velar', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'χ': {'place': 'uvular', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'minus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ʁ': {'place': 'uvular', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ħ': {'place': 'pharyngeal', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'minus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ʕ': {'place': 'pharyngeal', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'h': {'place': 'glottal', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'minus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ɦ': {'place': 'glottal', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ɬ': {'place': 'alveolar', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'minus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'plus', 'aspirated': 'minus'}, 'ɮ': {'place': 'alveolar', 'manner': 'fricative', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'plus', 'aspirated': 'minus'}, 'ʋ': {'place': 'labiodental', 'manner': 'approximant', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ɹ': {'place': 'alveolar', 'manner': 'approximant', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ɻ': {'place': 'retroflex', 'manner': 'approximant', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'plus', 'lateral': 'minus', 'aspirated': 'minus'}, 'j': {'place': 'palatal', 'manner': 'approximant', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'ɰ': {'place': 'velar', 'manner': 'approximant', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'l': {'place': 'alveolar', 'manner': 'approximant', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'plus', 'aspirated': 'minus'}, 'w': {'place': 'labiovelar', 'manner': 'approximant', 'syllabic': 'minus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'aspirated': 'minus'}, 'i': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'high', 'back': 'front', 'round': 'minus', 'long': 'minus', 'aspirated': 'minus'}, 'y': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'high', 'back': 'front', 'round': 'plus', 'long': 'minus', 'aspirated': 'minus'}, 'e': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'mid', 'back': 'front', 'round': 'minus', 'long': 'minus', 'aspirated': 'minus'}, 'E': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'mid', 'back': 'front', 'round': 'minus', 'long': 'plus', 'aspirated': 'minus'}, 'ø': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'mid', 'back': 'front', 'round': 'plus', 'long': 'minus', 'aspirated': 'minus'}, 'ɛ': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'mid', 'back': 'front', 'round': 'minus', 'long': 'minus', 'aspirated': 'minus'}, 'œ': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'mid', 'back': 'front', 'round': 'plus', 'long': 'minus', 'aspirated': 'minus'}, 'æ': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'low', 'back': 'front', 'round': 'minus', 'long': 'minus', 'aspirated': 'minus'}, 'a': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'low', 'back': 'front', 'round': 'minus', 'long': 'minus', 'aspirated': 'minus'}, 'A': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'low', 'back': 'front', 'round': 'minus', 'long': 'plus', 'aspirated': 'minus'}, 'ɨ': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'high', 'back': 'central', 'round': 'minus', 'long': 'minus', 'aspirated': 'minus'}, 'ʉ': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'high', 'back': 'central', 'round': 'plus', 'long': 'minus', 'aspirated': 'minus'}, 'ə': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'mid', 'back': 'central', 'round': 'minus', 'long': 'minus', 'aspirated': 'minus'}, 'u': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'high', 'back': 'back', 'round': 'plus', 'long': 'minus', 'aspirated': 'minus'}, 'U': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'high', 'back': 'back', 'round': 'plus', 'long': 'plus', 'aspirated': 'minus'}, 'o': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'mid', 'back': 'back', 'round': 'plus', 'long': 'minus', 'aspirated': 'minus'}, 'O': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'mid', 'back': 'back', 'round': 'plus', 'long': 'plus', 'aspirated': 'minus'}, 'ɔ': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'mid', 'back': 'back', 'round': 'plus', 'long': 'minus', 'aspirated': 'minus'}, 'ɒ': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'low', 'back': 'back', 'round': 'minus', 'long': 'minus', 'aspirated': 'minus'}, 'I': {'place': 'vowel', 'manner': 'vowel2', 'syllabic': 'plus', 'voice': 'plus', 'nasal': 'minus', 'retroflex': 'minus', 'lateral': 'minus', 'high': 'high', 'back': 'front', 'round': 'minus', 'long': 'plus', 'aspirated': 'minus'}}

def align(str1, str2, epsilon=0):
    if False:
        while True:
            i = 10
    '\n    Compute the alignment of two phonetic strings.\n\n    :param str str1: First string to be aligned\n    :param str str2: Second string to be aligned\n\n    :type epsilon: float (0.0 to 1.0)\n    :param epsilon: Adjusts threshold similarity score for near-optimal alignments\n\n    :rtype: list(list(tuple(str, str)))\n    :return: Alignment(s) of str1 and str2\n\n    (Kondrak 2002: 51)\n    '
    if np is None:
        raise ImportError('You need numpy in order to use the align function')
    assert 0.0 <= epsilon <= 1.0, 'Epsilon must be between 0.0 and 1.0.'
    m = len(str1)
    n = len(str2)
    S = np.zeros((m + 1, n + 1), dtype=float)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            edit1 = S[i - 1, j] + sigma_skip(str1[i - 1])
            edit2 = S[i, j - 1] + sigma_skip(str2[j - 1])
            edit3 = S[i - 1, j - 1] + sigma_sub(str1[i - 1], str2[j - 1])
            if i > 1:
                edit4 = S[i - 2, j - 1] + sigma_exp(str2[j - 1], str1[i - 2:i])
            else:
                edit4 = -inf
            if j > 1:
                edit5 = S[i - 1, j - 2] + sigma_exp(str1[i - 1], str2[j - 2:j])
            else:
                edit5 = -inf
            S[i, j] = max(edit1, edit2, edit3, edit4, edit5, 0)
    T = (1 - epsilon) * np.amax(S)
    alignments = []
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if S[i, j] >= T:
                alignments.append(_retrieve(i, j, 0, S, T, str1, str2, []))
    return alignments

def _retrieve(i, j, s, S, T, str1, str2, out):
    if False:
        return 10
    '\n    Retrieve the path through the similarity matrix S starting at (i, j).\n\n    :rtype: list(tuple(str, str))\n    :return: Alignment of str1 and str2\n    '
    if S[i, j] == 0:
        return out
    elif j > 1 and S[i - 1, j - 2] + sigma_exp(str1[i - 1], str2[j - 2:j]) + s >= T:
        out.insert(0, (str1[i - 1], str2[j - 2:j]))
        _retrieve(i - 1, j - 2, s + sigma_exp(str1[i - 1], str2[j - 2:j]), S, T, str1, str2, out)
    elif i > 1 and S[i - 2, j - 1] + sigma_exp(str2[j - 1], str1[i - 2:i]) + s >= T:
        out.insert(0, (str1[i - 2:i], str2[j - 1]))
        _retrieve(i - 2, j - 1, s + sigma_exp(str2[j - 1], str1[i - 2:i]), S, T, str1, str2, out)
    elif S[i, j - 1] + sigma_skip(str2[j - 1]) + s >= T:
        out.insert(0, ('-', str2[j - 1]))
        _retrieve(i, j - 1, s + sigma_skip(str2[j - 1]), S, T, str1, str2, out)
    elif S[i - 1, j] + sigma_skip(str1[i - 1]) + s >= T:
        out.insert(0, (str1[i - 1], '-'))
        _retrieve(i - 1, j, s + sigma_skip(str1[i - 1]), S, T, str1, str2, out)
    elif S[i - 1, j - 1] + sigma_sub(str1[i - 1], str2[j - 1]) + s >= T:
        out.insert(0, (str1[i - 1], str2[j - 1]))
        _retrieve(i - 1, j - 1, s + sigma_sub(str1[i - 1], str2[j - 1]), S, T, str1, str2, out)
    return out

def sigma_skip(p):
    if False:
        return 10
    '\n    Returns score of an indel of P.\n\n    (Kondrak 2002: 54)\n    '
    return C_skip

def sigma_sub(p, q):
    if False:
        i = 10
        return i + 15
    '\n    Returns score of a substitution of P with Q.\n\n    (Kondrak 2002: 54)\n    '
    return C_sub - delta(p, q) - V(p) - V(q)

def sigma_exp(p, q):
    if False:
        return 10
    '\n    Returns score of an expansion/compression.\n\n    (Kondrak 2002: 54)\n    '
    q1 = q[0]
    q2 = q[1]
    return C_exp - delta(p, q1) - delta(p, q2) - V(p) - max(V(q1), V(q2))

def delta(p, q):
    if False:
        return 10
    '\n    Return weighted sum of difference between P and Q.\n\n    (Kondrak 2002: 54)\n    '
    features = R(p, q)
    total = 0
    for f in features:
        total += diff(p, q, f) * salience[f]
    return total

def diff(p, q, f):
    if False:
        while True:
            i = 10
    '\n    Returns difference between phonetic segments P and Q for feature F.\n\n    (Kondrak 2002: 52, 54)\n    '
    (p_features, q_features) = (feature_matrix[p], feature_matrix[q])
    return abs(similarity_matrix[p_features[f]] - similarity_matrix[q_features[f]])

def R(p, q):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return relevant features for segment comparison.\n\n    (Kondrak 2002: 54)\n    '
    if p in consonants or q in consonants:
        return R_c
    return R_v

def V(p):
    if False:
        print('Hello World!')
    '\n    Return vowel weight if P is vowel.\n\n    (Kondrak 2002: 54)\n    '
    if p in consonants:
        return 0
    return C_vwl

def demo():
    if False:
        while True:
            i = 10
    "\n    A demonstration of the result of aligning phonetic sequences\n    used in Kondrak's (2002) dissertation.\n    "
    data = [pair.split(',') for pair in cognate_data.split('\n')]
    for pair in data:
        alignment = align(pair[0], pair[1])[0]
        alignment = [f'({a[0]}, {a[1]})' for a in alignment]
        alignment = ' '.join(alignment)
        print(f'{pair[0]} ~ {pair[1]} : {alignment}')
cognate_data = 'jo,ʒə\ntu,ty\nnosotros,nu\nkjen,ki\nke,kwa\ntodos,tu\nuna,ən\ndos,dø\ntres,trwa\nombre,om\narbol,arbrə\npluma,plym\nkabeθa,kap\nboka,buʃ\npje,pje\nkoraθon,kœr\nber,vwar\nbenir,vənir\ndeθir,dir\npobre,povrə\nðis,dIzes\nðæt,das\nwat,vas\nnat,nixt\nloŋ,laŋ\nmæn,man\nfleʃ,flajʃ\nbləd,blyt\nfeðər,fEdər\nhær,hAr\nir,Or\naj,awgə\nnowz,nAzə\nmawθ,munt\ntəŋ,tsuŋə\nfut,fys\nnij,knI\nhænd,hant\nhart,herts\nlivər,lEbər\nænd,ante\næt,ad\nblow,flAre\nir,awris\nijt,edere\nfiʃ,piʃkis\nflow,fluere\nstaɾ,stella\nful,plenus\ngræs,gramen\nhart,kordis\nhorn,korny\naj,ego\nnij,genU\nməðər,mAter\nmawntən,mons\nnejm,nomen\nnjuw,nowus\nwən,unus\nrawnd,rotundus\nsow,suere\nsit,sedere\nθrij,tres\ntuwθ,dentis\nθin,tenwis\nkinwawa,kenuaʔ\nnina,nenah\nnapewa,napɛw\nwapimini,wapemen\nnamesa,namɛʔs\nokimawa,okemaw\nʃiʃipa,seʔsep\nahkohkwa,ahkɛh\npematesiweni,pematesewen\nasenja,aʔsɛn'
if __name__ == '__main__':
    demo()