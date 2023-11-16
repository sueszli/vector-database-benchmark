class Solution(object):

    def depthSumInverse(self, nestedList):
        if False:
            return 10
        '\n        :type nestedList: List[NestedInteger]\n        :rtype: int\n        '

        def depthSumInverseHelper(list, depth, result):
            if False:
                print('Hello World!')
            if len(result) < depth + 1:
                result.append(0)
            if list.isInteger():
                result[depth] += list.getInteger()
            else:
                for l in list.getList():
                    depthSumInverseHelper(l, depth + 1, result)
        result = []
        for list in nestedList:
            depthSumInverseHelper(list, 0, result)
        sum = 0
        for i in reversed(xrange(len(result))):
            sum += result[i] * (len(result) - i)
        return sum