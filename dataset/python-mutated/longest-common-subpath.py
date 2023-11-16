class Solution(object):

    def longestCommonSubpath(self, n, paths):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type paths: List[List[int]]\n        :rtype: int\n        '

        def RabinKarp(arr, x):
            if False:
                while True:
                    i = 10
            hashes = tuple([reduce(lambda h, x: (h * p + x) % MOD, (arr[i] for i in xrange(x)), 0) for p in P])
            powers = [pow(p, x, MOD) for p in P]
            lookup = {hashes}
            for i in xrange(x, len(arr)):
                hashes = tuple([(hashes[j] * P[j] - arr[i - x] * powers[j] + arr[i]) % MOD for j in xrange(len(P))])
                lookup.add(hashes)
            return lookup

        def check(paths, x):
            if False:
                while True:
                    i = 10
            intersect = RabinKarp(paths[0], x)
            for i in xrange(1, len(paths)):
                intersect = set.intersection(intersect, RabinKarp(paths[i], x))
                if not intersect:
                    return False
            return True
        (MOD, P) = (10 ** 9 + 7, (113, 109))
        (left, right) = (1, min((len(p) for p in paths)))
        while left <= right:
            mid = left + (right - left) // 2
            if not check(paths, mid):
                right = mid - 1
            else:
                left = mid + 1
        return right

class Solution2(object):

    def longestCommonSubpath(self, n, paths):
        if False:
            return 10
        '\n        :type n: int\n        :type paths: List[List[int]]\n        :rtype: int\n        '

        def RabinKarp(arr, x):
            if False:
                i = 10
                return i + 15
            h = reduce(lambda h, x: (h * P + x) % MOD, (arr[i] for i in xrange(x)), 0)
            power = pow(P, x, MOD)
            lookup = {h}
            for i in xrange(x, len(arr)):
                h = (h * P - arr[i - x] * power + arr[i]) % MOD
                lookup.add(h)
            return lookup

        def check(paths, x):
            if False:
                while True:
                    i = 10
            intersect = RabinKarp(paths[0], x)
            for i in xrange(1, len(paths)):
                intersect = set.intersection(intersect, RabinKarp(paths[i], x))
                if not intersect:
                    return False
            return True
        (MOD, P) = (10 ** 11 + 19, max((x for p in paths for x in p)) + 1)
        (left, right) = (1, min((len(p) for p in paths)))
        while left <= right:
            mid = left + (right - left) // 2
            if not check(paths, mid):
                right = mid - 1
            else:
                left = mid + 1
        return right