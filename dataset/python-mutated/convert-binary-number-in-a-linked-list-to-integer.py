class ListNode(object):

    def __init__(self, x):
        if False:
            return 10
        self.val = x
        self.next = None

class Solution(object):

    def getDecimalValue(self, head):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type head: ListNode\n        :rtype: int\n        '
        result = 0
        while head:
            result = result * 2 + head.val
            head = head.next
        return result