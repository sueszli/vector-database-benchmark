import collections

class Solution(object):

    def numSquarefulPerms(self, A):
        if False:
            i = 10
            return i + 15
        '\n        :type A: List[int]\n        :rtype: int\n        '

        def dfs(candidate, x, left, count, result):
            if False:
                print('Hello World!')
            count[x] -= 1
            if left == 0:
                result[0] += 1
            for y in candidate[x]:
                if count[y]:
                    dfs(candidate, y, left - 1, count, result)
            count[x] += 1
        count = collections.Counter(A)
        candidate = {i: {j for j in count if int((i + j) ** 0.5) ** 2 == i + j} for i in count}
        result = [0]
        for x in count:
            dfs(candidate, x, len(A) - 1, count, result)
        return result[0]