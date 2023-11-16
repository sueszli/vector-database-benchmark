class Solution(object):

    def isPalindrome(self, head):
        if False:
            for i in range(10):
                print('nop')
        (reverse, fast) = (None, head)
        while fast and fast.next:
            fast = fast.next.next
            (head.next, reverse, head) = (reverse, head, head.next)
        tail = head.next if fast else head
        is_palindrome = True
        while reverse:
            is_palindrome = is_palindrome and reverse.val == tail.val
            (reverse.next, head, reverse) = (head, reverse, reverse.next)
            tail = tail.next
        return is_palindrome