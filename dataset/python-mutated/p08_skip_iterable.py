"""
Topic: 跳过可迭代对象开始部分
Desc : 
"""
from itertools import dropwhile
from itertools import islice

def skip_iter():
    if False:
        print('Hello World!')
    items = ['a', 'b', 'c', 1, 4, 10, 15]
    for x in islice(items, 3, None):
        print(x)
if __name__ == '__main__':
    skip_iter()