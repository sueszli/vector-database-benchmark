class ListNode(object):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.val = x
        self.next = None

class Solution(object):

    def nextLargerNodes(self, head):
        if False:
            return 10
        '\n        :type head: ListNode\n        :rtype: List[int]\n        '
        (result, stk) = ([], [])
        while head:
            while stk and stk[-1][1] < head.val:
                result[stk.pop()[0]] = head.val
            stk.append([len(result), head.val])
            result.append(0)
            head = head.next
        return result