"""
Calculate cosine similarity between given two 1d list.
Two list must have the same length.

Example:
cosine_similarity([1, 1, 1], [1, 2, -1])  # output : 0.47140452079103173
"""
import math

def _l2_distance(vec):
    if False:
        return 10
    '\n    Calculate l2 distance from two given vectors.\n    '
    norm = 0.0
    for element in vec:
        norm += element * element
    norm = math.sqrt(norm)
    return norm

def cosine_similarity(vec1, vec2):
    if False:
        while True:
            i = 10
    '\n    Calculate cosine similarity between given two vectors\n    :type vec1: list\n    :type vec2: list\n    '
    if len(vec1) != len(vec2):
        raise ValueError('The two vectors must be the same length. Got shape ' + str(len(vec1)) + ' and ' + str(len(vec2)))
    norm_a = _l2_distance(vec1)
    norm_b = _l2_distance(vec2)
    similarity = 0.0
    for (vec1_element, vec2_element) in zip(vec1, vec2):
        similarity += vec1_element * vec2_element
    similarity /= norm_a * norm_b
    return similarity