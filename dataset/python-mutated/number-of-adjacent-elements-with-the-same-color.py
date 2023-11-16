class Solution(object):

    def colorTheArray(self, n, queries):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type queries: List[List[int]]\n        :rtype: List[int]\n        '

        def update(i):
            if False:
                i = 10
                return i + 15
            if not nums[i]:
                return 0
            cnt = 0
            if i - 1 >= 0 and nums[i - 1] == nums[i]:
                cnt += 1
            if i + 1 < n and nums[i + 1] == nums[i]:
                cnt += 1
            return cnt
        nums = [0] * n
        result = [0] * len(queries)
        curr = 0
        for (idx, (i, c)) in enumerate(queries):
            curr -= update(i)
            nums[i] = c
            curr += update(i)
            result[idx] = curr
        return result