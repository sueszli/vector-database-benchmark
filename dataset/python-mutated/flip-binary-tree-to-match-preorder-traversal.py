class TreeNode(object):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def flipMatchVoyage(self, root, voyage):
        if False:
            i = 10
            return i + 15
        '\n        :type root: TreeNode\n        :type voyage: List[int]\n        :rtype: List[int]\n        '

        def dfs(root, voyage, i, result):
            if False:
                for i in range(10):
                    print('nop')
            if not root:
                return True
            if root.val != voyage[i[0]]:
                return False
            i[0] += 1
            if root.left and root.left.val != voyage[i[0]]:
                result.append(root.val)
                return dfs(root.right, voyage, i, result) and dfs(root.left, voyage, i, result)
            return dfs(root.left, voyage, i, result) and dfs(root.right, voyage, i, result)
        result = []
        return result if dfs(root, voyage, [0], result) else [-1]