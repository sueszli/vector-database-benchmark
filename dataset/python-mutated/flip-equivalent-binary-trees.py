class TreeNode(object):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.val = x
        self.left = None
        self.right = None
import collections

class Solution(object):

    def flipEquiv(self, root1, root2):
        if False:
            return 10
        '\n        :type root1: TreeNode\n        :type root2: TreeNode\n        :rtype: bool\n        '
        (dq1, dq2) = (collections.deque([root1]), collections.deque([root2]))
        while dq1 and dq2:
            (node1, node2) = (dq1.pop(), dq2.pop())
            if not node1 and (not node2):
                continue
            if not node1 or not node2 or node1.val != node2.val:
                return False
            if not node1.left and (not node2.right) or (node1.left and node2.right and (node1.left.val == node2.right.val)):
                dq1.extend([node1.right, node1.left])
            else:
                dq1.extend([node1.left, node1.right])
            dq2.extend([node2.left, node2.right])
        return not dq1 and (not dq2)

class Solution2(object):

    def flipEquiv(self, root1, root2):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type root1: TreeNode\n        :type root2: TreeNode\n        :rtype: bool\n        '
        (stk1, stk2) = ([root1], [root2])
        while stk1 and stk2:
            (node1, node2) = (stk1.pop(), stk2.pop())
            if not node1 and (not node2):
                continue
            if not node1 or not node2 or node1.val != node2.val:
                return False
            if not node1.left and (not node2.right) or (node1.left and node2.right and (node1.left.val == node2.right.val)):
                stk1.extend([node1.right, node1.left])
            else:
                stk1.extend([node1.left, node1.right])
            stk2.extend([node2.left, node2.right])
        return not stk1 and (not stk2)

class Solution3(object):

    def flipEquiv(self, root1, root2):
        if False:
            print('Hello World!')
        '\n        :type root1: TreeNode\n        :type root2: TreeNode\n        :rtype: bool\n        '
        if not root1 and (not root2):
            return True
        if not root1 or not root2 or root1.val != root2.val:
            return False
        return self.flipEquiv(root1.left, root2.left) and self.flipEquiv(root1.right, root2.right) or (self.flipEquiv(root1.left, root2.right) and self.flipEquiv(root1.right, root2.left))