import unittest
from test import support
from test.support import warnings_helper
import os
import sys
import importlib
if support.check_sanitizer(address=True, memory=True):
    raise unittest.SkipTest('workaround ASAN build issues on loading tests like tk or crypt')

class NoAll(RuntimeError):
    pass

class FailedImport(RuntimeError):
    pass

class AllTest(unittest.TestCase):

    def check_all(self, modname):
        if False:
            return 10
        names = {}
        with warnings_helper.check_warnings(('.* (module|package)', DeprecationWarning), ('.* (module|package)', PendingDeprecationWarning), ('', ResourceWarning), quiet=True):
            try:
                exec('import %s; %s' % (modname, modname), names)
                if importlib.is_lazy_imports_enabled():
                    exec('%s' % modname)
            except:
                raise FailedImport(modname)
        if not hasattr(sys.modules[modname], '__all__'):
            raise NoAll(modname)
        names = {}
        with self.subTest(module=modname):
            with warnings_helper.check_warnings(('', DeprecationWarning), ('', ResourceWarning), quiet=True):
                try:
                    exec('from %s import *' % modname, names)
                except Exception as e:
                    self.fail('__all__ failure in {}: {}: {}'.format(modname, e.__class__.__name__, e))
                if '__builtins__' in names:
                    del names['__builtins__']
                if '__annotations__' in names:
                    del names['__annotations__']
                if '__warningregistry__' in names:
                    del names['__warningregistry__']
                keys = set(names)
                all_list = sys.modules[modname].__all__
                all_set = set(all_list)
                self.assertCountEqual(all_set, all_list, 'in module {}'.format(modname))
                self.assertEqual(keys, all_set, 'in module {}'.format(modname))

    def walk_modules(self, basedir, modpath):
        if False:
            while True:
                i = 10
        for fn in sorted(os.listdir(basedir)):
            path = os.path.join(basedir, fn)
            if os.path.isdir(path):
                pkg_init = os.path.join(path, '__init__.py')
                if os.path.exists(pkg_init):
                    yield (pkg_init, modpath + fn)
                    for (p, m) in self.walk_modules(path, modpath + fn + '.'):
                        yield (p, m)
                continue
            if not fn.endswith('.py') or fn == '__init__.py':
                continue
            yield (path, modpath + fn[:-3])

    def test_all(self):
        if False:
            while True:
                i = 10
        denylist = set(['__future__'])
        if not sys.platform.startswith('java'):
            import _socket
        ignored = []
        failed_imports = []
        lib_dir = os.path.dirname(os.path.dirname(__file__))
        for (path, modname) in self.walk_modules(lib_dir, ''):
            m = modname
            denied = False
            while m:
                if m in denylist:
                    denied = True
                    break
                m = m.rpartition('.')[0]
            if denied:
                continue
            if support.verbose:
                print(modname)
            try:
                with open(path, 'rb') as f:
                    if b'__all__' not in f.read():
                        raise NoAll(modname)
                    self.check_all(modname)
            except NoAll:
                ignored.append(modname)
            except FailedImport:
                failed_imports.append(modname)
        if support.verbose:
            print('Following modules have no __all__ and have been ignored:', ignored)
            print('Following modules failed to be imported:', failed_imports)
if __name__ == '__main__':
    unittest.main()