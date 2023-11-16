"""
迭代使用的是循环结构。
递归使用的是选择结构。
"""
from __future__ import print_function

def calculate(l):
    if False:
        while True:
            i = 10
    if len(l) <= 1:
        return l[0]
    value = calculate(l[1:])
    return 10 ** (len(l) - 1) * l[0] + value

def calculate2(l):
    if False:
        print('Hello World!')
    result = 0
    while len(l) >= 1:
        result += 10 ** (len(l) - 1) * l[0]
        l = l[1:]
    return result
l1 = [1, 2, 3]
l2 = [4, 5]
sum = 0
result = calculate(l1) + calculate(l2)
print(result)