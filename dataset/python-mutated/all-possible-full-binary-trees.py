class TreeNode(object):

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.__memo = {1: [TreeNode(0)]}

    def allPossibleFBT(self, N):
        if False:
            while True:
                i = 10
        '\n        :type N: int\n        :rtype: List[TreeNode]\n        '
        if N % 2 == 0:
            return []
        if N not in self.__memo:
            result = []
            for i in xrange(N):
                for left in self.allPossibleFBT(i):
                    for right in self.allPossibleFBT(N - 1 - i):
                        node = TreeNode(0)
                        node.left = left
                        node.right = right
                        result.append(node)
            self.__memo[N] = result
        return self.__memo[N]