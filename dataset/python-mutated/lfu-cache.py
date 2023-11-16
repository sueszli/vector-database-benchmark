import collections

class LFUCache(object):

    def __init__(self, capacity):
        if False:
            print('Hello World!')
        '\n        :type capacity: int\n        '
        self.__capa = capacity
        self.__size = 0
        self.__min_freq = float('inf')
        self.__freq_to_nodes = collections.defaultdict(collections.OrderedDict)
        self.__key_to_freq = {}

    def get(self, key):
        if False:
            print('Hello World!')
        '\n        :type key: int\n        :rtype: int\n        '
        if key not in self.__key_to_freq:
            return -1
        value = self.__freq_to_nodes[self.__key_to_freq[key]][key]
        self.__update(key, value)
        return value

    def put(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type key: int\n        :type value: int\n        :rtype: void\n        '
        if self.__capa <= 0:
            return
        if key not in self.__key_to_freq and self.__size == self.__capa:
            del self.__key_to_freq[self.__freq_to_nodes[self.__min_freq].popitem(last=False)[0]]
            if not self.__freq_to_nodes[self.__min_freq]:
                del self.__freq_to_nodes[self.__min_freq]
            self.__size -= 1
        self.__update(key, value)

    def __update(self, key, value):
        if False:
            for i in range(10):
                print('nop')
        freq = 0
        if key in self.__key_to_freq:
            freq = self.__key_to_freq[key]
            del self.__freq_to_nodes[freq][key]
            if not self.__freq_to_nodes[freq]:
                del self.__freq_to_nodes[freq]
                if self.__min_freq == freq:
                    self.__min_freq += 1
            self.__size -= 1
        freq += 1
        self.__min_freq = min(self.__min_freq, freq)
        self.__key_to_freq[key] = freq
        self.__freq_to_nodes[freq][key] = value
        self.__size += 1
import collections

class ListNode(object):

    def __init__(self, key, value, freq):
        if False:
            return 10
        self.key = key
        self.val = value
        self.freq = freq
        self.next = None
        self.prev = None

class LinkedList(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.head = None
        self.tail = None

    def append(self, node):
        if False:
            return 10
        (node.next, node.prev) = (None, None)
        if self.head is None:
            self.head = node
        else:
            self.tail.next = node
            node.prev = self.tail
        self.tail = node

    def delete(self, node):
        if False:
            i = 10
            return i + 15
        if node.prev:
            node.prev.next = node.next
        else:
            self.head = node.next
        if node.next:
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        (node.next, node.prev) = (None, None)

class LFUCache2(object):

    def __init__(self, capacity):
        if False:
            while True:
                i = 10
        '\n        :type capacity: int\n        '
        self.__capa = capacity
        self.__size = 0
        self.__min_freq = float('inf')
        self.__freq_to_nodes = collections.defaultdict(LinkedList)
        self.__key_to_node = {}

    def get(self, key):
        if False:
            while True:
                i = 10
        '\n        :type key: int\n        :rtype: int\n        '
        if key not in self.__key_to_node:
            return -1
        value = self.__key_to_node[key].val
        self.__update(key, value)
        return value

    def put(self, key, value):
        if False:
            print('Hello World!')
        '\n        :type key: int\n        :type value: int\n        :rtype: void\n        '
        if self.__capa <= 0:
            return
        if key not in self.__key_to_node and self.__size == self.__capa:
            del self.__key_to_node[self.__freq_to_nodes[self.__min_freq].head.key]
            self.__freq_to_nodes[self.__min_freq].delete(self.__freq_to_nodes[self.__min_freq].head)
            if not self.__freq_to_nodes[self.__min_freq].head:
                del self.__freq_to_nodes[self.__min_freq]
            self.__size -= 1
        self.__update(key, value)

    def __update(self, key, value):
        if False:
            print('Hello World!')
        freq = 0
        if key in self.__key_to_node:
            old_node = self.__key_to_node[key]
            freq = old_node.freq
            self.__freq_to_nodes[freq].delete(old_node)
            if not self.__freq_to_nodes[freq].head:
                del self.__freq_to_nodes[freq]
                if self.__min_freq == freq:
                    self.__min_freq += 1
            self.__size -= 1
        freq += 1
        self.__min_freq = min(self.__min_freq, freq)
        self.__key_to_node[key] = ListNode(key, value, freq)
        self.__freq_to_nodes[freq].append(self.__key_to_node[key])
        self.__size += 1