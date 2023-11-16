class Node(object):

    def __init__(self, val=0, left=None, right=None, next=None):
        if False:
            while True:
                i = 10
        self.val = val
        self.left = left
        self.right = right
        self.next = next

class Solution(object):

    def connect(self, root):
        if False:
            print('Hello World!')
        head = root
        pre = Node(0)
        cur = pre
        while root:
            while root:
                if root.left:
                    cur.next = root.left
                    cur = cur.next
                if root.right:
                    cur.next = root.right
                    cur = cur.next
                root = root.next
            (root, cur) = (pre.next, pre)
            cur.next = None
        return head