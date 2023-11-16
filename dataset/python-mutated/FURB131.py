from typing import Dict, List
names = {'key': 'value'}
nums = [1, 2, 3]
x = 42
y = 'hello'
del nums[:]
del names[:]
del x, nums[:]
del y, names[:], x

def yes_one(x: list[int]):
    if False:
        i = 10
        return i + 15
    del x[:]

def yes_two(x: dict[int, str]):
    if False:
        print('Hello World!')
    del x[:]

def yes_three(x: List[int]):
    if False:
        i = 10
        return i + 15
    del x[:]

def yes_four(x: Dict[int, str]):
    if False:
        for i in range(10):
            print('nop')
    del x[:]

def yes_five(x: Dict[int, str]):
    if False:
        return 10
    del x[:]
    x = 1
del names['key']
del nums[0]
x = 1
del x
del nums[1:2]
del nums[:2]
del nums[1:]
del nums[::2]

def no_one(param):
    if False:
        for i in range(10):
            print('nop')
    del param[:]