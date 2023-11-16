import self
if self._lazy_imports:
    self.skipTest('Test relevant only when running with global lazy imports disabled')
import importlib

class Checker:
    matches = 0

    def filter(self, name):
        if False:
            while True:
                i = 10
        return name == 'test.lazyimports.data.excluding.bar'

    def __contains__(self, name):
        if False:
            for i in range(10):
                print('nop')
        if self.filter(name):
            self.matches += 1
            return True
        return False
checker = Checker()
importlib.set_lazy_imports(excluding=checker)
from test.lazyimports.data.excluding import foo
self.assertTrue(importlib.is_lazy_import(foo.__dict__, 'Foo'))
from test.lazyimports.data.excluding import bar
self.assertFalse(importlib.is_lazy_import(bar.__dict__, 'Bar'))
self.assertEqual(foo.Foo, 'Foo')
self.assertFalse(importlib.is_lazy_import(foo.__dict__, 'Foo'))
self.assertEqual(checker.matches, 1)