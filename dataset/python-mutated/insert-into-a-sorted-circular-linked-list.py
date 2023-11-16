class Node(object):

    def __init__(self, val, next):
        if False:
            while True:
                i = 10
        self.val = val
        self.next = next

class Solution(object):

    def insert(self, head, insertVal):
        if False:
            return 10
        '\n        :type head: Node\n        :type insertVal: int\n        :rtype: Node\n        '

        def insertAfter(node, val):
            if False:
                return 10
            node.next = Node(val, node.next)
        if not head:
            node = Node(insertVal, None)
            node.next = node
            return node
        curr = head
        while True:
            if curr.val < curr.next.val:
                if curr.val <= insertVal and insertVal <= curr.next.val:
                    insertAfter(curr, insertVal)
                    break
            elif curr.val > curr.next.val:
                if curr.val <= insertVal or insertVal <= curr.next.val:
                    insertAfter(curr, insertVal)
                    break
            elif curr.next == head:
                insertAfter(curr, insertVal)
                break
            curr = curr.next
        return head