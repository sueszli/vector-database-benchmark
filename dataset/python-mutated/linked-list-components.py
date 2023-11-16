class ListNode(object):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.val = x
        self.next = None

class Solution(object):

    def numComponents(self, head, G):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type head: ListNode\n        :type G: List[int]\n        :rtype: int\n        '
        lookup = set(G)
        dummy = ListNode(-1)
        dummy.next = head
        curr = dummy
        result = 0
        while curr and curr.next:
            if curr.val not in lookup and curr.next.val in lookup:
                result += 1
            curr = curr.next
        return result