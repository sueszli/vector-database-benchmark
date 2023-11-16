class Solution(object):

    def findLeaves(self, root):
        if False:
            print('Hello World!')
        '\n        :type root: TreeNode\n        :rtype: List[List[int]]\n        '

        def findLeavesHelper(node, result):
            if False:
                print('Hello World!')
            if not node:
                return -1
            level = 1 + max(findLeavesHelper(node.left, result), findLeavesHelper(node.right, result))
            if len(result) < level + 1:
                result.append([])
            result[level].append(node.val)
            return level
        result = []
        findLeavesHelper(root, result)
        return result