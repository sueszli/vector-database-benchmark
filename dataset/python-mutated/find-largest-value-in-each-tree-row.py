class Solution(object):

    def largestValues(self, root):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :rtype: List[int]\n        '

        def largestValuesHelper(root, depth, result):
            if False:
                i = 10
                return i + 15
            if not root:
                return
            if depth == len(result):
                result.append(root.val)
            else:
                result[depth] = max(result[depth], root.val)
            largestValuesHelper(root.left, depth + 1, result)
            largestValuesHelper(root.right, depth + 1, result)
        result = []
        largestValuesHelper(root, 0, result)
        return result

class Solution2(object):

    def largestValues(self, root):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :rtype: List[int]\n        '
        result = []
        curr = [root]
        while any(curr):
            result.append(max((node.val for node in curr)))
            curr = [child for node in curr for child in (node.left, node.right) if child]
        return result