class Solution(object):

    def getRow(self, rowIndex):
        if False:
            print('Hello World!')
        result = [0] * (rowIndex + 1)
        for i in xrange(rowIndex + 1):
            old = result[0] = 1
            for j in xrange(1, i + 1):
                (old, result[j]) = (result[j], old + result[j])
        return result

    def getRow2(self, rowIndex):
        if False:
            i = 10
            return i + 15
        '\n        :type rowIndex: int\n        :rtype: List[int]\n        '
        row = [1]
        for _ in range(rowIndex):
            row = [x + y for (x, y) in zip([0] + row, row + [0])]
        return row

    def getRow3(self, rowIndex):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type rowIndex: int\n        :rtype: List[int]\n        '
        if rowIndex == 0:
            return [1]
        res = [1, 1]

        def add(nums):
            if False:
                return 10
            res = nums[:1]
            for (i, j) in enumerate(nums):
                if i < len(nums) - 1:
                    res += [nums[i] + nums[i + 1]]
            res += nums[:1]
            return res
        while res[1] < rowIndex:
            res = add(res)
        return res

class Solution2(object):

    def getRow(self, rowIndex):
        if False:
            return 10
        result = [1]
        for i in range(1, rowIndex + 1):
            result = [1] + [result[j - 1] + result[j] for j in xrange(1, i)] + [1]
        return result