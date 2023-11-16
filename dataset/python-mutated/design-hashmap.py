class ListNode(object):

    def __init__(self, key, val):
        if False:
            return 10
        self.val = val
        self.key = key
        self.next = None
        self.prev = None

class LinkedList(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.head = None
        self.tail = None

    def insert(self, node):
        if False:
            i = 10
            return i + 15
        (node.next, node.prev) = (None, None)
        if self.head is None:
            self.head = node
        else:
            self.tail.next = node
            node.prev = self.tail
        self.tail = node

    def delete(self, node):
        if False:
            print('Hello World!')
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        (node.next, node.prev) = (None, None)

    def find(self, key):
        if False:
            i = 10
            return i + 15
        curr = self.head
        while curr:
            if curr.key == key:
                break
            curr = curr.next
        return curr

class MyHashMap(object):

    def __init__(self):
        if False:
            print('Hello World!')
        '\n        Initialize your data structure here.\n        '
        self.__data = [LinkedList() for _ in xrange(10000)]

    def put(self, key, value):
        if False:
            while True:
                i = 10
        '\n        value will always be positive.\n        :type key: int\n        :type value: int\n        :rtype: void\n        '
        l = self.__data[key % len(self.__data)]
        node = l.find(key)
        if node:
            node.val = value
        else:
            l.insert(ListNode(key, value))

    def get(self, key):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key\n        :type key: int\n        :rtype: int\n        '
        l = self.__data[key % len(self.__data)]
        node = l.find(key)
        if node:
            return node.val
        else:
            return -1

    def remove(self, key):
        if False:
            for i in range(10):
                print('nop')
        '\n        Removes the mapping of the specified value key if this map contains a mapping for the key\n        :type key: int\n        :rtype: void\n        '
        l = self.__data[key % len(self.__data)]
        node = l.find(key)
        if node:
            l.delete(node)