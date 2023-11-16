class node:

    def __init__(self, data):
        if False:
            for i in range(10):
                print('nop')
        self.data = data
        self.next = None

class linkedList:

    def __init__(self):
        if False:
            print('Hello World!')
        self.head = None

    def push(self, newdata):
        if False:
            for i in range(10):
                print('nop')
        newNode = node(newdata)
        newNode.next = self.head
        self.head = newNode

    def insertAfter(self, prevnode, newdata):
        if False:
            print('Hello World!')
        if prevnode is None:
            return
        newnode = node(newdata)
        newnode.next = prevnode.next
        prevnode.next = newnode

    def append(self, newdata):
        if False:
            for i in range(10):
                print('nop')
        newnode = node(newdata)
        if self.head is None:
            self.head = newnode
            return
        last = self.head
        while last.next:
            last = last.next
        last.next = newnode

    def printList(self):
        if False:
            i = 10
            return i + 15
        temp = self.head
        while temp:
            print(temp.data, end=' ')
            temp = temp.next
if __name__ == '__main__':
    llist = linkedList()
    llist.append(6)
    llist.push(7)
    llist.push(1)
    llist.append(4)
    llist.insertAfter(llist.head.next, 8)
    llist.printList()