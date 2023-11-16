"""
Topic: 函数默认参数
Desc : 
"""

def spam(a, b=42):
    if False:
        while True:
            i = 10
    print(a, b)
spam(1)
spam(1, 2)
_no_value = object()

def spam(a, b=_no_value):
    if False:
        while True:
            i = 10
    if b is _no_value:
        print('No b value supplied')