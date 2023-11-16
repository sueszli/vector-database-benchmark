import itertools

class Solution(object):

    def maximumRequests(self, n, requests):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type requests: List[List[int]]\n        :rtype: int\n        '
        for k in reversed(xrange(1, len(requests) + 1)):
            for c in itertools.combinations(xrange(len(requests)), k):
                change = [0] * n
                for i in c:
                    change[requests[i][0]] -= 1
                    change[requests[i][1]] += 1
                if all((c == 0 for c in change)):
                    return k
        return 0

class Solution2(object):

    def maximumRequests(self, n, requests):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type requests: List[List[int]]\n        :rtype: int\n        '

        def evaluate(n, requests, mask):
            if False:
                while True:
                    i = 10
            change = [0] * n
            (base, count) = (1, 0)
            for i in xrange(len(requests)):
                if base & mask:
                    change[requests[i][0]] -= 1
                    change[requests[i][1]] += 1
                    count += 1
                base <<= 1
            return count if all((c == 0 for c in change)) else 0
        return max((evaluate(n, requests, i) for i in xrange(1 << len(requests))))