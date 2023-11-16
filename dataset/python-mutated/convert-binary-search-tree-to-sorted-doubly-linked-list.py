class Node(object):

    def __init__(self, val, left, right):
        if False:
            return 10
        self.val = val
        self.left = left
        self.right = right

class Solution(object):

    def treeToDoublyList(self, root):
        if False:
            print('Hello World!')
        '\n        :type root: Node\n        :rtype: Node\n        '
        if not root:
            return None
        (left_head, left_tail, right_head, right_tail) = (root, root, root, root)
        if root.left:
            left_head = self.treeToDoublyList(root.left)
            left_tail = left_head.left
        if root.right:
            right_head = self.treeToDoublyList(root.right)
            right_tail = right_head.left
        (left_tail.right, right_head.left) = (root, root)
        (root.left, root.right) = (left_tail, right_head)
        (left_head.left, right_tail.right) = (right_tail, left_head)
        return left_head