from collections import deque

class Vector2D(object):

    def __init__(self, vec2d):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize your data structure here.\n        :type vec2d: List[List[int]]\n        '
        self.stack = deque(((len(v), iter(v)) for v in vec2d if v))

    def next(self):
        if False:
            return 10
        '\n        :rtype: int\n        '
        (length, iterator) = self.stack.popleft()
        if length > 1:
            self.stack.appendleft((length - 1, iterator))
        return next(iterator)

    def hasNext(self):
        if False:
            return 10
        '\n        :rtype: bool\n        '
        return bool(self.stack)