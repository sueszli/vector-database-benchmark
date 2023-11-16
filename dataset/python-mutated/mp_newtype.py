from multiprocessing import freeze_support
from multiprocessing.managers import BaseManager, BaseProxy
import operator

class Foo:

    def f(self):
        if False:
            i = 10
            return i + 15
        print('you called Foo.f()')

    def g(self):
        if False:
            print('Hello World!')
        print('you called Foo.g()')

    def _h(self):
        if False:
            while True:
                i = 10
        print('you called Foo._h()')

def baz():
    if False:
        return 10
    for i in range(10):
        yield (i * i)

class GeneratorProxy(BaseProxy):
    _exposed_ = ['__next__']

    def __iter__(self):
        if False:
            while True:
                i = 10
        return self

    def __next__(self):
        if False:
            while True:
                i = 10
        return self._callmethod('__next__')

def get_operator_module():
    if False:
        return 10
    return operator

class MyManager(BaseManager):
    pass
MyManager.register('Foo1', Foo)
MyManager.register('Foo2', Foo, exposed=('g', '_h'))
MyManager.register('baz', baz, proxytype=GeneratorProxy)
MyManager.register('operator', get_operator_module)

def test():
    if False:
        return 10
    manager = MyManager()
    manager.start()
    print('-' * 20)
    f1 = manager.Foo1()
    f1.f()
    f1.g()
    assert not hasattr(f1, '_h')
    assert sorted(f1._exposed_) == sorted(['f', 'g'])
    print('-' * 20)
    f2 = manager.Foo2()
    f2.g()
    f2._h()
    assert not hasattr(f2, 'f')
    assert sorted(f2._exposed_) == sorted(['g', '_h'])
    print('-' * 20)
    it = manager.baz()
    for i in it:
        print('<%d>' % i, end=' ')
    print()
    print('-' * 20)
    op = manager.operator()
    print('op.add(23, 45) =', op.add(23, 45))
    print('op.pow(2, 94) =', op.pow(2, 94))
    print('op._exposed_ =', op._exposed_)
if __name__ == '__main__':
    freeze_support()
    test()