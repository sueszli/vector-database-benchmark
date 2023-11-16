import collections

class Solution(object):

    def checkEqualTree(self, root):
        if False:
            i = 10
            return i + 15
        '\n        :type root: TreeNode\n        :rtype: bool\n        '

        def getSumHelper(node, lookup):
            if False:
                for i in range(10):
                    print('nop')
            if not node:
                return 0
            total = node.val + getSumHelper(node.left, lookup) + getSumHelper(node.right, lookup)
            lookup[total] += 1
            return total
        lookup = collections.defaultdict(int)
        total = getSumHelper(root, lookup)
        if total == 0:
            return lookup[total] > 1
        return total % 2 == 0 and total / 2 in lookup