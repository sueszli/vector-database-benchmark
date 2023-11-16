class Node(object):
    """
    double linked list node
    """

    def __init__(self, value, keys):
        if False:
            return 10
        self.value = value
        self.keys = keys
        self.prev = None
        self.next = None

class LinkedList(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        (self.head, self.tail) = (Node(0, set()), Node(0, set()))
        (self.head.next, self.tail.prev) = (self.tail, self.head)

    def insert(self, pos, node):
        if False:
            for i in range(10):
                print('nop')
        (node.prev, node.next) = (pos.prev, pos)
        (pos.prev.next, pos.prev) = (node, node)
        return node

    def erase(self, node):
        if False:
            print('Hello World!')
        (node.prev.next, node.next.prev) = (node.next, node.prev)
        del node

    def empty(self):
        if False:
            for i in range(10):
                print('nop')
        return self.head.next is self.tail

    def begin(self):
        if False:
            print('Hello World!')
        return self.head.next

    def end(self):
        if False:
            while True:
                i = 10
        return self.tail

    def front(self):
        if False:
            return 10
        return self.head.next

    def back(self):
        if False:
            print('Hello World!')
        return self.tail.prev

class AllOne(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        '\n        Initialize your data structure here.\n        '
        self.bucket_of_key = {}
        self.buckets = LinkedList()

    def inc(self, key):
        if False:
            print('Hello World!')
        '\n        Inserts a new key <Key> with value 1. Or increments an existing key by 1.\n        :type key: str\n        :rtype: void\n        '
        if key not in self.bucket_of_key:
            self.bucket_of_key[key] = self.buckets.insert(self.buckets.begin(), Node(0, set([key])))
        (bucket, next_bucket) = (self.bucket_of_key[key], self.bucket_of_key[key].next)
        if next_bucket is self.buckets.end() or next_bucket.value > bucket.value + 1:
            next_bucket = self.buckets.insert(next_bucket, Node(bucket.value + 1, set()))
        next_bucket.keys.add(key)
        self.bucket_of_key[key] = next_bucket
        bucket.keys.remove(key)
        if not bucket.keys:
            self.buckets.erase(bucket)

    def dec(self, key):
        if False:
            for i in range(10):
                print('nop')
        "\n        Decrements an existing key by 1. If Key's value is 1, remove it from the data structure.\n        :type key: str\n        :rtype: void\n        "
        if key not in self.bucket_of_key:
            return
        (bucket, prev_bucket) = (self.bucket_of_key[key], self.bucket_of_key[key].prev)
        self.bucket_of_key.pop(key, None)
        if bucket.value > 1:
            if bucket is self.buckets.begin() or prev_bucket.value < bucket.value - 1:
                prev_bucket = self.buckets.insert(bucket, Node(bucket.value - 1, set()))
            prev_bucket.keys.add(key)
            self.bucket_of_key[key] = prev_bucket
        bucket.keys.remove(key)
        if not bucket.keys:
            self.buckets.erase(bucket)

    def getMaxKey(self):
        if False:
            print('Hello World!')
        '\n        Returns one of the keys with maximal value.\n        :rtype: str\n        '
        if self.buckets.empty():
            return ''
        return iter(self.buckets.back().keys).next()

    def getMinKey(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns one of the keys with Minimal value.\n        :rtype: str\n        '
        if self.buckets.empty():
            return ''
        return iter(self.buckets.front().keys).next()