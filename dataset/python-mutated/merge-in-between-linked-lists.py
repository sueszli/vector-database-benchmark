class ListNode(object):

    def __init__(self, val=0, next=None):
        if False:
            print('Hello World!')
        pass

class Solution(object):

    def mergeInBetween(self, list1, a, b, list2):
        if False:
            print('Hello World!')
        '\n        :type list1: ListNode\n        :type a: int\n        :type b: int\n        :type list2: ListNode\n        :rtype: ListNode\n        '
        (prev_first, last) = (None, list1)
        for i in xrange(b):
            if i == a - 1:
                prev_first = last
            last = last.next
        prev_first.next = list2
        while list2.next:
            list2 = list2.next
        list2.next = last.next
        last.next = None
        return list1