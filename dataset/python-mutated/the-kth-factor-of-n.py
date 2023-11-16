class Solution(object):

    def kthFactor(self, n, k):
        if False:
            return 10
        '\n        :type n: int\n        :type k: int\n        :rtype: int\n        '

        def kth_factor(n, k=0):
            if False:
                return 10
            mid = None
            i = 1
            while i * i <= n:
                if not n % i:
                    mid = i
                    k -= 1
                    if not k:
                        break
                i += 1
            return (mid, -k)
        (mid, count) = kth_factor(n)
        total = 2 * count - (mid * mid == n)
        if k > total:
            return -1
        result = kth_factor(n, k if k <= count else total - (k - 1))[0]
        return result if k <= count else n // result

class Solution2(object):

    def kthFactor(self, n, k):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type k: int\n        :rtype: int\n        '
        result = []
        i = 1
        while i * i <= n:
            if not n % i:
                if i * i != n:
                    result.append(i)
                k -= 1
                if not k:
                    return i
            i += 1
        return -1 if k > len(result) else n // result[-k]