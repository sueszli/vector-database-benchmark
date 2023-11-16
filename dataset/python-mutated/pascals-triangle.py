class Solution(object):

    def generate(self, numRows):
        if False:
            i = 10
            return i + 15
        result = []
        for i in xrange(numRows):
            result.append([])
            for j in xrange(i + 1):
                if j in (0, i):
                    result[i].append(1)
                else:
                    result[i].append(result[i - 1][j - 1] + result[i - 1][j])
        return result

    def generate2(self, numRows):
        if False:
            return 10
        if not numRows:
            return []
        res = [[1]]
        for i in range(1, numRows):
            res += [map(lambda x, y: x + y, res[-1] + [0], [0] + res[-1])]
        return res[:numRows]

    def generate3(self, numRows):
        if False:
            return 10
        '\n        :type numRows: int\n        :rtype: List[List[int]]\n        '
        if numRows == 0:
            return []
        if numRows == 1:
            return [[1]]
        res = [[1], [1, 1]]

        def add(nums):
            if False:
                for i in range(10):
                    print('nop')
            res = nums[:1]
            for (i, j) in enumerate(nums):
                if i < len(nums) - 1:
                    res += [nums[i] + nums[i + 1]]
            res += nums[:1]
            return res
        while len(res) < numRows:
            res.extend([add(res[-1])])
        return res