class Solution(object):

    def splitListToParts(self, root, k):
        if False:
            return 10
        '\n        :type root: ListNode\n        :type k: int\n        :rtype: List[ListNode]\n        '
        n = 0
        curr = root
        while curr:
            curr = curr.next
            n += 1
        (width, remainder) = divmod(n, k)
        result = []
        curr = root
        for i in xrange(k):
            head = curr
            for j in xrange(width - 1 + int(i < remainder)):
                if curr:
                    curr = curr.next
            if curr:
                (curr.next, curr) = (None, curr.next)
            result.append(head)
        return result