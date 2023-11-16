class Node:

    def __init__(self, data):
        if False:
            print('Hello World!')
        self.data = data
        self.next = None

class Solution:

    def display(self, head):
        if False:
            while True:
                i = 10
        current = head
        while current:
            print(current.data, end=' ')
            current = current.next

    def insert(self, head, data):
        if False:
            print('Hello World!')
        if head == None:
            head = Node(data)
        elif head.next == None:
            head.next = Node(data)
        else:
            self.insert(head.next, data)
        return head
mylist = Solution()
T = int(input())
head = None
for i in range(T):
    data = int(input())
    head = mylist.insert(head, data)
mylist.display(head)