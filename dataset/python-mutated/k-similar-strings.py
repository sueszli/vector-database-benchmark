import collections

class Solution(object):

    def kSimilarity(self, A, B):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type A: str\n        :type B: str\n        :rtype: int\n        '

        def neighbors(s, B):
            if False:
                for i in range(10):
                    print('nop')
            for (i, c) in enumerate(s):
                if c != B[i]:
                    break
            t = list(s)
            for j in xrange(i + 1, len(s)):
                if t[j] == B[i]:
                    (t[i], t[j]) = (t[j], t[i])
                    yield ''.join(t)
                    (t[j], t[i]) = (t[i], t[j])
        q = collections.deque([A])
        lookup = set()
        result = 0
        while q:
            for _ in xrange(len(q)):
                s = q.popleft()
                if s == B:
                    return result
                for t in neighbors(s, B):
                    if t not in lookup:
                        lookup.add(t)
                        q.append(t)
            result += 1