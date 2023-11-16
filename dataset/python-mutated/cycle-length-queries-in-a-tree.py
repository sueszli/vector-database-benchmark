class Solution(object):

    def cycleLengthQueries(self, n, queries):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type queries: List[List[int]]\n        :rtype: List[int]\n        '
        result = []
        for (x, y) in queries:
            cnt = 1
            while x != y:
                if x > y:
                    (x, y) = (y, x)
                y //= 2
                cnt += 1
            result.append(cnt)
        return result