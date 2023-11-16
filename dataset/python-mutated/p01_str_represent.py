"""
Topic: 实例的字符串显示
Desc : 
"""

class Pair:

    def __init__(self, x, y):
        if False:
            while True:
                i = 10
        self.x = x
        self.y = y

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Pair({0.x!r}, {0.y!r})'.format(self)

    def __str__(self):
        if False:
            print('Hello World!')
        return '({0.x!s}, {0.y!s})'.format(self)
p = Pair(3, 4)
print(p)