class Solution(object):

    def hIndex(self, citations):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type citations: List[int]\n        :rtype: int\n        '
        n = len(citations)
        count = [0] * (n + 1)
        for x in citations:
            if x >= n:
                count[n] += 1
            else:
                count[x] += 1
        h = 0
        for i in reversed(xrange(0, n + 1)):
            h += count[i]
            if h >= i:
                return i
        return h

class Solution2(object):

    def hIndex(self, citations):
        if False:
            while True:
                i = 10
        '\n        :type citations: List[int]\n        :rtype: int\n        '
        citations.sort(reverse=True)
        h = 0
        for x in citations:
            if x >= h + 1:
                h += 1
            else:
                break
        return h

class Solution3(object):

    def hIndex(self, citations):
        if False:
            print('Hello World!')
        '\n        :type citations: List[int]\n        :rtype: int\n        '
        return sum((x >= i + 1 for (i, x) in enumerate(sorted(citations, reverse=True))))