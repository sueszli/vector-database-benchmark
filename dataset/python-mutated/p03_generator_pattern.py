"""
Topic: 使用生成器创建新的迭代模式
Desc : 
"""

def frange(start, stop, increment):
    if False:
        while True:
            i = 10
    x = start
    while x < stop:
        yield x
        x += increment

def countdown(n):
    if False:
        i = 10
        return i + 15
    print('Starting to count from', n)
    while n > 0:
        yield n
        n -= 1
    print('Done')

def gen_pattern():
    if False:
        print('Hello World!')
    for n in frange(0, 4, 0.5):
        print(n)
    print(list(frange(0, 1, 0.125)))
    c = countdown(3)
    print(next(c))
    print(next(c))
    print(next(c))
    print(next(c))
if __name__ == '__main__':
    gen_pattern()