class ListNode:

    def __init__(self, val=0, next=None):
        if False:
            for i in range(10):
                print('nop')
        self.val = val
        self.next = next

class LinkedList:

    def __init__(self):
        if False:
            return 10
        self.head = None

    def create(self, data):
        if False:
            while True:
                i = 10
        self.head = ListNode(0)
        cur = self.head
        for i in range(len(data)):
            node = ListNode(data[i])
            cur.next = node
            cur = cur.next

    def length(self):
        if False:
            i = 10
            return i + 15
        count = 0
        cur = self.head
        while cur:
            count += 1
            cur = cur.next
        return count

    def find(self, val):
        if False:
            i = 10
            return i + 15
        cur = self.head
        while cur:
            if val == cur.val:
                return cur
            cur = cur.next
        return None

    def insertFront(self, val):
        if False:
            return 10
        node = ListNode(val)
        node.next = self.head
        self.head = node

    def insertRear(self, val):
        if False:
            for i in range(10):
                print('nop')
        node = ListNode(val)
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = node

    def insertInside(self, index, val):
        if False:
            print('Hello World!')
        count = 0
        cur = self.head
        while cur and count < index - 1:
            count += 1
            cur = cur.next
        if not cur:
            return 'Error'
        node = ListNode(val)
        node.next = cur.next
        cur.next = node

    def change(self, index, val):
        if False:
            return 10
        count = 0
        cur = self.head
        while cur and count < index:
            count += 1
            cur = cur.next
        if not cur:
            return 'Error'
        cur.val = val

    def removeFront(self):
        if False:
            print('Hello World!')
        if self.head:
            self.head = self.head.next

    def removeRear(self):
        if False:
            return 10
        if not self.head or not self.head.next:
            return 'Error'
        cur = self.head
        while cur.next.next:
            cur = cur.next
        cur.next = None

    def removeInside(self, index):
        if False:
            for i in range(10):
                print('nop')
        count = 0
        cur = self.head
        while cur.next and count < index - 1:
            count += 1
            cur = cur.next
        if not cur:
            return 'Error'
        del_node = cur.next
        cur.next = del_node.next