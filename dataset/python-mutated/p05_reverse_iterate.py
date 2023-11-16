"""
Topic: 方向迭代
Desc : 
"""

def reverse_iterate():
    if False:
        for i in range(10):
            print('nop')
    a = [1, 2, 3, 4]
    for x in reversed(a):
        print(x)
    for rr in reversed(Countdown(30)):
        print(rr)
    for rr in Countdown(30):
        print(rr)

class Countdown:

    def __init__(self, start):
        if False:
            while True:
                i = 10
        self.start = start

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        n = self.start
        while n > 0:
            yield n
            n -= 1

    def __reversed__(self):
        if False:
            print('Hello World!')
        n = 1
        while n <= self.start:
            yield n
            n += 1
if __name__ == '__main__':
    reverse_iterate()