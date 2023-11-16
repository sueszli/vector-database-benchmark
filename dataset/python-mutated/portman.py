from collections import defaultdict, Counter
from typing import List, Tuple, Set, Dict, Any
Word = str

class Wordset(set):
    """A set of words."""
Step = Tuple[int, str]
(OVERLAP, WORD) = (0, 1)
Path = List[Step]
Bridge = (int, Step, ...)
(EXCESS, STEPS) = (0, slice(1, None))
W = Wordset(open('wordlist.asc').read().split())

def portman(P: Path) -> Word:
    if False:
        print('Hello World!')
    'Compute the portmantout string S from the path P.'
    return ''.join((word[overlap:] for (overlap, word) in P))

def natalie(W: Wordset, start=None) -> Path:
    if False:
        return 10
    'Return a portmantout path containing all words in W.'
    precompute(W)
    word = start or first(W.unused)
    used(W, word)
    P = [(0, word)]
    while W.unused:
        steps = unused_step(W, word) or bridging_steps(W, word)
        for (overlap, word) in steps:
            P.append((overlap, word))
            used(W, word)
    return P

def unused_step(W: Wordset, prev_word: Word) -> List[Step]:
    if False:
        for i in range(10):
            print('nop')
    'Return [(overlap, unused_word)] or [].'
    for suf in suffixes(prev_word):
        for unused_word in W.startswith.get(suf, ()):
            overlap = len(suf)
            return [(overlap, unused_word)]
    return []

def bridging_steps(W: Wordset, prev_word: Word) -> List[Step]:
    if False:
        i = 10
        return i + 15
    'The steps from the shortest bridge that bridges \n    from a suffix of prev_word to a prefix of an unused word.'
    bridge = min((W.bridges[suf][pre] for suf in suffixes(prev_word) if suf in W.bridges for pre in W.bridges[suf] if W.startswith[pre]))
    return bridge[STEPS]

def precompute(W):
    if False:
        for i in range(10):
            print('nop')
    'Precompute and cache data structures for W. The .subwords and .bridges\n    data structures are static and only need to be computed once; .unused and\n    .startswith are dynamic and must be recomputed on each call to `natalie`.'
    if not hasattr(W, 'subwords') or not hasattr(W, 'bridges'):
        W.subwords = subwords(W)
        W.bridges = build_bridges(W)
    W.unused = W - W.subwords
    W.startswith = compute_startswith(W.unused)

def used(W, word):
    if False:
        print('Hello World!')
    'Remove word from `W.unused` and, for each prefix, from `W.startswith[pre]`.'
    assert word in W, f'used "{word}", which is not in the word set'
    if word in W.unused:
        W.unused.remove(word)
        for pre in prefixes(word):
            W.startswith[pre].remove(word)
            if not W.startswith[pre]:
                del W.startswith[pre]

def first(iterable, default=None):
    if False:
        return 10
    return next(iter(iterable), default)

def multimap(pairs) -> Dict[Any, set]:
    if False:
        return 10
    'Given (key, val) pairs, make a dict of {key: {val,...}}.'
    result = defaultdict(set)
    for (key, val) in pairs:
        result[key].add(val)
    return result

def compute_startswith(words) -> Dict[str, Set[Word]]:
    if False:
        i = 10
        return i + 15
    "A dict mapping a prefix to all the words it starts:\n    {'somet': {'something', 'sometimes'},...}."
    return multimap(((pre, w) for w in words for pre in prefixes(w)))

def subwords(W: Wordset) -> Set[str]:
    if False:
        print('Hello World!')
    'All the words in W that are subparts of some other word.'
    return {subword for w in W for subword in subparts(w) & W}

def suffixes(word) -> List[str]:
    if False:
        return 10
    'All non-empty proper suffixes of word, longest first.'
    return [word[i:] for i in range(1, len(word))]

def prefixes(word) -> List[str]:
    if False:
        while True:
            i = 10
    'All non-empty proper prefixes of word.'
    return [word[:i] for i in range(1, len(word))]

def subparts(word) -> Set[str]:
    if False:
        i = 10
        return i + 15
    'All non-empty proper substrings of word'
    return {word[i:j] for i in range(len(word)) for j in range(i + 1, len(word) + (i > 0))}

def splits(word) -> List[Tuple[int, str, str]]:
    if False:
        while True:
            i = 10
    'A sequence of (excess, pre, suf) tuples.'
    return [(excess, word[:i], word[i + excess:]) for excess in range(len(word) - 1) for i in range(1, len(word) - excess)]

def try_bridge(bridges, pre, suf, excess, word, step2=None):
    if False:
        print('Hello World!')
    'Store a new bridge if it has less excess than the previous bridges[pre][suf].'
    if suf not in bridges[pre] or excess < bridges[pre][suf][EXCESS]:
        bridge = (excess, (len(pre), word))
        if step2:
            bridge += (step2,)
        bridges[pre][suf] = bridge

def build_bridges(W: Wordset, maxlen=5, end='qujvz'):
    if False:
        return 10
    "A table of bridges[pre][suf] == (excess, (overlap, word)), e.g.\n    bridges['ar']['c'] == (0, (2, 'arc'))."
    bridges = defaultdict(dict)
    shortwords = [w for w in W if len(w) <= maxlen + (w[-1] in end)]
    shortstartswith = compute_startswith(shortwords)
    for word in shortwords:
        for (excess, pre, suf) in splits(word):
            try_bridge(bridges, pre, suf, excess, word)
    for word1 in shortwords:
        for suf in suffixes(word1):
            for word2 in shortstartswith[suf]:
                excess = len(word1) + len(word2) - len(suf) - 2
                (A, B) = (word1[0], word2[-1])
                if A != B:
                    step2 = (len(suf), word2)
                    try_bridge(bridges, A, B, excess, word1, step2)
    return bridges
if __name__ == '__main__':
    W = Wordset(open('wordlist.asc').read().split())
    print(portman(natalie(W)))