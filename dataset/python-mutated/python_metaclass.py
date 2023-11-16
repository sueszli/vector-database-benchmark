"""
python_metaclass.py by xianhu
"""

class Foo(object):

    def hello(self):
        if False:
            return 10
        print('hello world!')
        return
foo = Foo()
print(type(foo))
print(type(foo.hello))
print(type(Foo))
temp = Foo
Foo.var = 11
print(Foo)

def init(self, name):
    if False:
        for i in range(10):
            print('nop')
    self.name = name
    return

def hello(self):
    if False:
        while True:
            i = 10
    print('hello %s' % self.name)
    return
Foo = type('Foo', (object,), {'__init__': init, 'hello': hello, 'cls_var': 10})
foo = Foo('xianhu')
print(foo.hello())
print(Foo.cls_var)
print(foo.__class__)
print(Foo.__class__)
print(type.__class__)

class Author(type):

    def __new__(mcs, name, bases, dict):
        if False:
            return 10
        dict['author'] = 'xianhu'
        return super(Author, mcs).__new__(mcs, name, bases, dict)

class Foo(object, metaclass=Author):
    pass
foo = Foo()
print(foo.author)