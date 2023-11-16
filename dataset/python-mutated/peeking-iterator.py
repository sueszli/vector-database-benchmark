class PeekingIterator(object):

    def __init__(self, iterator):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize your data structure here.\n        :type iterator: Iterator\n        '
        self.iterator = iterator
        self.val_ = None
        self.has_next_ = iterator.hasNext()
        self.has_peeked_ = False

    def peek(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the next element in the iteration without advancing the iterator.\n        :rtype: int\n        '
        if not self.has_peeked_:
            self.has_peeked_ = True
            self.val_ = self.iterator.next()
        return self.val_

    def next(self):
        if False:
            while True:
                i = 10
        '\n        :rtype: int\n        '
        self.val_ = self.peek()
        self.has_peeked_ = False
        self.has_next_ = self.iterator.hasNext()
        return self.val_

    def hasNext(self):
        if False:
            i = 10
            return i + 15
        '\n        :rtype: bool\n        '
        return self.has_next_