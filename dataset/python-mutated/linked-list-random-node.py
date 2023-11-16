from random import randint

class Solution(object):

    def __init__(self, head):
        if False:
            for i in range(10):
                print('nop')
        "\n        @param head The linked list's head. Note that the head is guanranteed to be not null, so it contains at least one node.\n        :type head: ListNode\n        "
        self.__head = head

    def getRandom(self):
        if False:
            while True:
                i = 10
        "\n        Returns a random node's value.\n        :rtype: int\n        "
        reservoir = -1
        (curr, n) = (self.__head, 0)
        while curr:
            reservoir = curr.val if randint(1, n + 1) == 1 else reservoir
            (curr, n) = (curr.next, n + 1)
        return reservoir