class Solution(object):

    def insertGreatestCommonDivisors(self, head):
        if False:
            while True:
                i = 10
        '\n        :type head: Optional[ListNode]\n        :rtype: Optional[ListNode]\n        '

        def gcd(a, b):
            if False:
                for i in range(10):
                    print('nop')
            while b:
                (a, b) = (b, a % b)
            return a
        curr = head
        while curr.next:
            curr.next = ListNode(gcd(curr.val, curr.next.val), curr.next)
            curr = curr.next.next
        return head