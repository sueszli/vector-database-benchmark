class TreeNode(object):

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.val = x
        self.left = None
        self.right = None
        self.next = None

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        if self is None:
            return 'Nil'
        else:
            return '{} -> {}'.format(self.val, repr(self.next))

class Solution(object):

    def connect(self, root):
        if False:
            print('Hello World!')
        head = root
        while head:
            cur = head
            while cur and cur.left:
                cur.left.next = cur.right
                if cur.next:
                    cur.right.next = cur.next.left
                cur = cur.next
            head = head.left

class Solution2(object):

    def connect(self, root):
        if False:
            print('Hello World!')
        if root is None:
            return
        if root.left:
            root.left.next = root.right
        if root.right and root.next:
            root.right.next = root.next.left
        self.connect(root.left)
        self.connect(root.right)