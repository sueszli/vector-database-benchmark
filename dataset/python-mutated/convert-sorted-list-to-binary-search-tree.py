class TreeNode(object):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.val = x
        self.left = None
        self.right = None

class ListNode(object):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.val = x
        self.next = None

class Solution(object):
    head = None

    def sortedListToBST(self, head):
        if False:
            for i in range(10):
                print('nop')
        (current, length) = (head, 0)
        while current is not None:
            (current, length) = (current.next, length + 1)
        self.head = head
        return self.sortedListToBSTRecu(0, length)

    def sortedListToBSTRecu(self, start, end):
        if False:
            while True:
                i = 10
        if start == end:
            return None
        mid = start + (end - start) / 2
        left = self.sortedListToBSTRecu(start, mid)
        current = TreeNode(self.head.val)
        current.left = left
        self.head = self.head.next
        current.right = self.sortedListToBSTRecu(mid + 1, end)
        return current