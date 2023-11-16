class Solution(object):

    def depthSum(self, nestedList):
        if False:
            while True:
                i = 10
        '\n        :type nestedList: List[NestedInteger]\n        :rtype: int\n        '

        def depthSumHelper(nestedList, depth):
            if False:
                return 10
            res = 0
            for l in nestedList:
                if l.isInteger():
                    res += l.getInteger() * depth
                else:
                    res += depthSumHelper(l.getList(), depth + 1)
            return res
        return depthSumHelper(nestedList, 1)