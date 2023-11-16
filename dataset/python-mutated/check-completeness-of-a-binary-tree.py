class TreeNode(object):

    def __init__(self, x):
        if False:
            print('Hello World!')
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def isCompleteTree(self, root):
        if False:
            return 10
        '\n        :type root: TreeNode\n        :rtype: bool\n        '
        end = False
        current = [root]
        while current:
            next_level = []
            for node in current:
                if not node:
                    end = True
                    continue
                if end:
                    return False
                next_level.append(node.left)
                next_level.append(node.right)
            current = next_level
        return True

class Solution2(object):

    def isCompleteTree(self, root):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root: TreeNode\n        :rtype: bool\n        '
        (prev_level, current) = ([], [(root, 1)])
        count = 0
        while current:
            count += len(current)
            next_level = []
            for (node, v) in current:
                if not node:
                    continue
                next_level.append((node.left, 2 * v))
                next_level.append((node.right, 2 * v + 1))
            (prev_level, current) = (current, next_level)
        return prev_level[-1][1] == count