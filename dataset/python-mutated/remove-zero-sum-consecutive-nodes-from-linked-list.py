import collections

class ListNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.next = None

class Solution(object):

    def removeZeroSumSublists(self, head):
        if False:
            while True:
                i = 10
        '\n        :type head: ListNode\n        :rtype: ListNode\n        '
        curr = dummy = ListNode(0)
        dummy.next = head
        prefix = 0
        lookup = collections.OrderedDict()
        while curr:
            prefix += curr.val
            node = lookup.get(prefix, curr)
            while prefix in lookup:
                lookup.popitem()
            lookup[prefix] = node
            node.next = curr.next
            curr = curr.next
        return dummy.next