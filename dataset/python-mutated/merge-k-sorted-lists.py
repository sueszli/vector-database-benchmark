class ListNode(object):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.val = x
        self.next = None

    def __repr__(self):
        if False:
            while True:
                i = 10
        if self:
            return '{} -> {}'.format(self.val, self.next)

class Solution(object):

    def mergeKLists(self, lists):
        if False:
            while True:
                i = 10
        '\n        :type lists: List[ListNode]\n        :rtype: ListNode\n        '

        def mergeTwoLists(l1, l2):
            if False:
                return 10
            curr = dummy = ListNode(0)
            while l1 and l2:
                if l1.val < l2.val:
                    curr.next = l1
                    l1 = l1.next
                else:
                    curr.next = l2
                    l2 = l2.next
                curr = curr.next
            curr.next = l1 or l2
            return dummy.next
        if not lists:
            return None
        (left, right) = (0, len(lists) - 1)
        while right > 0:
            lists[left] = mergeTwoLists(lists[left], lists[right])
            left += 1
            right -= 1
            if left >= right:
                left = 0
        return lists[0]

class Solution2(object):

    def mergeKLists(self, lists):
        if False:
            while True:
                i = 10

        def mergeTwoLists(l1, l2):
            if False:
                print('Hello World!')
            curr = dummy = ListNode(0)
            while l1 and l2:
                if l1.val < l2.val:
                    curr.next = l1
                    l1 = l1.next
                else:
                    curr.next = l2
                    l2 = l2.next
                curr = curr.next
            curr.next = l1 or l2
            return dummy.next

        def mergeKListsHelper(lists, begin, end):
            if False:
                for i in range(10):
                    print('nop')
            if begin > end:
                return None
            if begin == end:
                return lists[begin]
            return mergeTwoLists(mergeKListsHelper(lists, begin, (begin + end) / 2), mergeKListsHelper(lists, (begin + end) / 2 + 1, end))
        return mergeKListsHelper(lists, 0, len(lists) - 1)
import heapq

class Solution3(object):

    def mergeKLists(self, lists):
        if False:
            print('Hello World!')
        dummy = ListNode(0)
        current = dummy
        heap = []
        for sorted_list in lists:
            if sorted_list:
                heapq.heappush(heap, (sorted_list.val, sorted_list))
        while heap:
            smallest = heapq.heappop(heap)[1]
            current.next = smallest
            current = current.next
            if smallest.next:
                heapq.heappush(heap, (smallest.next.val, smallest.next))
        return dummy.next