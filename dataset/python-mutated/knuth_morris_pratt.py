from typing import Sequence, List

def knuth_morris_pratt(text: Sequence, pattern: Sequence) -> List[int]:
    if False:
        return 10
    "\n    Given two strings text and pattern, return the list of start indexes in text that matches with the pattern\n    using knuth_morris_pratt algorithm.\n\n    Args:\n        text: Text to search\n        pattern: Pattern to search in the text\n    Returns:\n        List of indices of patterns found\n\n    Example:\n        >>> knuth_morris_pratt('hello there hero!', 'he')\n        [0, 7, 12]\n\n    If idx is in the list, text[idx : idx + M] matches with pattern.\n    Time complexity of the algorithm is O(N+M), with N and M the length of text and pattern, respectively.\n    "
    n = len(text)
    m = len(pattern)
    pi = [0 for i in range(m)]
    i = 0
    j = 0
    for i in range(1, m):
        while j and pattern[i] != pattern[j]:
            j = pi[j - 1]
        if pattern[i] == pattern[j]:
            j += 1
            pi[i] = j
    j = 0
    ret = []
    for i in range(n):
        while j and text[i] != pattern[j]:
            j = pi[j - 1]
        if text[i] == pattern[j]:
            j += 1
            if j == m:
                ret.append(i - m + 1)
                j = pi[j - 1]
    return ret