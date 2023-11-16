class ListNode(object):

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.val = x
        self.next = None

class Solution(object):

    def getIntersectionNode(self, headA, headB):
        if False:
            while True:
                i = 10
        (curA, curB) = (headA, headB)
        while curA != curB:
            curA = curA.next if curA else headB
            curB = curB.next if curB else headA
        return curA