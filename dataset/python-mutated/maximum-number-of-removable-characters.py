class Solution(object):

    def maximumRemovals(self, s, p, removable):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :type p: str\n        :type removable: List[int]\n        :rtype: int\n        '

        def check(s, p, removable, x):
            if False:
                return 10
            lookup = set((removable[i] for i in xrange(x)))
            j = 0
            for i in xrange(len(s)):
                if i in lookup or s[i] != p[j]:
                    continue
                j += 1
                if j == len(p):
                    return True
            return False
        (left, right) = (0, len(removable))
        while left <= right:
            mid = left + (right - left) // 2
            if not check(s, p, removable, mid):
                right = mid - 1
            else:
                left = mid + 1
        return right

class Solution2(object):

    def maximumRemovals(self, s, p, removable):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :type p: str\n        :type removable: List[int]\n        :rtype: int\n        '

        def check(s, p, lookup, x):
            if False:
                for i in range(10):
                    print('nop')
            j = 0
            for i in xrange(len(s)):
                if lookup[i] <= x or s[i] != p[j]:
                    continue
                j += 1
                if j == len(p):
                    return True
            return False
        lookup = [float('inf')] * len(s)
        for (i, r) in enumerate(removable):
            lookup[r] = i + 1
        (left, right) = (0, len(removable))
        while left <= right:
            mid = left + (right - left) // 2
            if not check(s, p, lookup, mid):
                right = mid - 1
            else:
                left = mid + 1
        return right