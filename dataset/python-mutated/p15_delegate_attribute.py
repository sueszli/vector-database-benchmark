"""
Topic: 属性访问代理
Desc : 
"""

class A:

    def spam(self, x):
        if False:
            return 10
        pass

    def foo(self):
        if False:
            return 10
        pass

class B1:
    """简单的代理"""

    def __init__(self):
        if False:
            return 10
        self._a = A()

    def spam(self, x):
        if False:
            return 10
        return self._a.spam(x)

    def foo(self):
        if False:
            while True:
                i = 10
        return self._a.foo()

    def bar(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class B2:
    """使用__getattr__的代理，代理方法比较多时候"""

    def __init__(self):
        if False:
            while True:
                i = 10
        self._a = A()

    def bar(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        '这个方法在访问的attribute不存在的时候被调用\n        the __getattr__() method is actually a fallback method\n        that only gets called when an attribute is not found'
        return getattr(self._a, name)
b = B2()
b.bar()
b.spam(42)

class Proxy:

    def __init__(self, obj):
        if False:
            return 10
        self._obj = obj

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        print('getattr:', name)
        return getattr(self._obj, name)

    def __setattr__(self, name, value):
        if False:
            print('Hello World!')
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            print('setattr:', name, value)
            setattr(self._obj, name, value)

    def __delattr__(self, name):
        if False:
            return 10
        if name.startswith('_'):
            super().__delattr__(name)
        else:
            print('delattr:', name)
            delattr(self._obj, name)

class Spam:

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.x = x

    def bar(self, y):
        if False:
            print('Hello World!')
        print('Spam.bar:', self.x, y)
s = Spam(2)
p = Proxy(s)
print(p.x)
p.bar(3)
p.x = 37

class ListLike:
    """__getattr__对于双下划线开始和结尾的方法是不能用的，需要一个个去重定义"""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._items = []

    def __getattr__(self, name):
        if False:
            return 10
        return getattr(self._items, name)

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self._items)

    def __getitem__(self, index):
        if False:
            return 10
        return self._items[index]

    def __setitem__(self, index, value):
        if False:
            return 10
        self._items[index] = value

    def __delitem__(self, index):
        if False:
            for i in range(10):
                print('nop')
        del self._items[index]
a = ListLike()
a.append(2)
a.insert(0, 1)
a.sort()
print(len(a))

class A:

    def spam(self, x):
        if False:
            print('Hello World!')
        print('A.spam', x)

    def foo(self):
        if False:
            return 10
        print('A.foo')

class B(A):

    def spam(self, x):
        if False:
            for i in range(10):
                print('nop')
        print('B.spam')
        super().spam(x)

    def bar(self):
        if False:
            i = 10
            return i + 15
        print('B.bar')

class A:

    def spam(self, x):
        if False:
            for i in range(10):
                print('nop')
        print('A.spam', x)

    def foo(self):
        if False:
            print('Hello World!')
        print('A.foo')

class B:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._a = A()

    def spam(self, x):
        if False:
            i = 10
            return i + 15
        print('B.spam', x)
        self._a.spam(x)

    def bar(self):
        if False:
            i = 10
            return i + 15
        print('B.bar')

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        return getattr(self._a, name)