import collections

class Solution(object):

    def numMatchingSubseq(self, S, words):
        if False:
            i = 10
            return i + 15
        '\n        :type S: str\n        :type words: List[str]\n        :rtype: int\n        '
        waiting = collections.defaultdict(list)
        for word in words:
            it = iter(word)
            waiting[next(it, None)].append(it)
        for c in S:
            for it in waiting.pop(c, ()):
                waiting[next(it, None)].append(it)
        return len(waiting[None])