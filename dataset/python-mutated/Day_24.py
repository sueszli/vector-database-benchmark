class Node:

    def __init__(self, data):
        if False:
            return 10
        self.data = data
        self.next = None

class Solution:

    def insert(self, head, data):
        if False:
            while True:
                i = 10
        p = Node(data)
        if head == None:
            head = p
        elif head.next == None:
            head.next = p
        else:
            start = head
            while start.next != None:
                start = start.next
            start.next = p
        return head

    def display(self, head):
        if False:
            return 10
        current = head
        while current:
            print(current.data, end=' ')
            current = current.next

    def removeDuplicates(self, head):
        if False:
            i = 10
            return i + 15
        if head == None:
            return head
        fptr = head.next
        sptr = head
        ha = {}
        while fptr != None:
            if sptr.data not in ha:
                ha[sptr.data] = True
            if fptr.data in ha:
                sptr.next = fptr.next
                fptr = fptr.next
                continue
            sptr = fptr
            fptr = fptr.next
        return head
mylist = Solution()
T = int(input())
head = None
for i in range(T):
    data = int(input())
    head = mylist.insert(head, data)
head = mylist.removeDuplicates(head)
mylist.display(head)