class Solution(object):

    def widthOfBinaryTree(self, root):
        if False:
            i = 10
            return i + 15
        '\n        :type root: TreeNode\n        :rtype: int\n        '

        def dfs(node, i, depth, leftmosts):
            if False:
                return 10
            if not node:
                return 0
            if depth >= len(leftmosts):
                leftmosts.append(i)
            return max(i - leftmosts[depth] + 1, dfs(node.left, i * 2, depth + 1, leftmosts), dfs(node.right, i * 2 + 1, depth + 1, leftmosts))
        leftmosts = []
        return dfs(root, 1, 0, leftmosts)