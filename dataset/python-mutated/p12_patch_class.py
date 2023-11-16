"""
Topic: 使用类装饰器来增强类功能
Desc : 
"""

def log_getattribute(cls):
    if False:
        return 10
    orig_getattribute = cls.__getattribute__

    def new_getattribute(self, name):
        if False:
            print('Hello World!')
        print('getting:', name)
        return orig_getattribute(self, name)
    cls.__getattribute__ = new_getattribute
    return cls

@log_getattribute
class A:

    def __init__(self, x):
        if False:
            return 10
        self.x = x

    def spam(self):
        if False:
            print('Hello World!')
        pass