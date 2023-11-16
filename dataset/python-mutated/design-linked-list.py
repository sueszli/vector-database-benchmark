class Node(object):

    def __init__(self, value):
        if False:
            print('Hello World!')
        self.val = value
        self.next = self.prev = None

class MyLinkedList(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        '\n        Initialize your data structure here.\n        '
        self.__head = self.__tail = Node(-1)
        self.__head.next = self.__tail
        self.__tail.prev = self.__head
        self.__size = 0

    def get(self, index):
        if False:
            i = 10
            return i + 15
        '\n        Get the value of the index-th node in the linked list. If the index is invalid, return -1.\n        :type index: int\n        :rtype: int\n        '
        if 0 <= index <= self.__size // 2:
            return self.__forward(0, index, self.__head.next).val
        elif self.__size // 2 < index < self.__size:
            return self.__backward(self.__size, index, self.__tail).val
        return -1

    def addAtHead(self, val):
        if False:
            print('Hello World!')
        '\n        Add a node of value val before the first element of the linked list.\n        After the insertion, the new node will be the first node of the linked list.\n        :type val: int\n        :rtype: void\n        '
        self.__add(self.__head, val)

    def addAtTail(self, val):
        if False:
            i = 10
            return i + 15
        '\n        Append a node of value val to the last element of the linked list.\n        :type val: int\n        :rtype: void\n        '
        self.__add(self.__tail.prev, val)

    def addAtIndex(self, index, val):
        if False:
            while True:
                i = 10
        '\n        Add a node of value val before the index-th node in the linked list.\n        If index equals to the length of linked list,\n        the node will be appended to the end of linked list.\n        If index is greater than the length, the node will not be inserted.\n        :type index: int\n        :type val: int\n        :rtype: void\n        '
        if 0 <= index <= self.__size // 2:
            self.__add(self.__forward(0, index, self.__head.next).prev, val)
        elif self.__size // 2 < index <= self.__size:
            self.__add(self.__backward(self.__size, index, self.__tail).prev, val)

    def deleteAtIndex(self, index):
        if False:
            print('Hello World!')
        '\n        Delete the index-th node in the linked list, if the index is valid.\n        :type index: int\n        :rtype: void\n        '
        if 0 <= index <= self.__size // 2:
            self.__remove(self.__forward(0, index, self.__head.next))
        elif self.__size // 2 < index < self.__size:
            self.__remove(self.__backward(self.__size, index, self.__tail))

    def __add(self, preNode, val):
        if False:
            for i in range(10):
                print('nop')
        node = Node(val)
        node.prev = preNode
        node.next = preNode.next
        node.prev.next = node.next.prev = node
        self.__size += 1

    def __remove(self, node):
        if False:
            for i in range(10):
                print('nop')
        node.prev.next = node.next
        node.next.prev = node.prev
        self.__size -= 1

    def __forward(self, start, end, curr):
        if False:
            for i in range(10):
                print('nop')
        while start != end:
            start += 1
            curr = curr.next
        return curr

    def __backward(self, start, end, curr):
        if False:
            return 10
        while start != end:
            start -= 1
            curr = curr.prev
        return curr