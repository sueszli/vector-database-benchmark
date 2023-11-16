"""
Topic: 解除包装器
Desc : 
"""
from functools import wraps

def decorator1(func):
    if False:
        for i in range(10):
            print('nop')

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        print('Decorator 1')
        return func(*args, **kwargs)
    return wrapper

def decorator2(func):
    if False:
        print('Hello World!')

    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            return 10
        print('Decorator 2')
        return func(*args, **kwargs)
    return wrapper

@decorator1
@decorator2
def add(x, y):
    if False:
        print('Hello World!')
    return x + y
add(1, 2)
print('-----------------')
print(add.__wrapped__(2, 3))