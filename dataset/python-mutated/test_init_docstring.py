import importlib
import inspect
import pkgutil
import unittest
import chainer
from chainer import testing

def get_init_doc(klass):
    if False:
        for i in range(10):
            print('nop')
    for attr in inspect.classify_class_attrs(klass):
        if attr.name == '__init__':
            if attr.defining_class is klass:
                return attr.object.__doc__
            else:
                return None
    return None

class TestInitDocstring(unittest.TestCase):
    """Make sure classes do not have a docstring in their __init__ method."""

    def check_init_docstring(self, mod, errors):
        if False:
            print('Hello World!')
        for (name, value) in inspect.getmembers(mod):
            if not inspect.isclass(value):
                continue
            if 'chainer' not in value.__module__:
                continue
            init_doc = get_init_doc(value)
            if init_doc == object.__init__.__doc__:
                continue
            if init_doc is not None:
                errors.append((mod, value, init_doc))

    def test_init_docstring_empty(self):
        if False:
            return 10
        errors = []
        root = chainer.__path__
        for (loader, modname, ispkg) in pkgutil.walk_packages(root, 'chainer.'):
            if '_pb2' in modname:
                continue
            try:
                mod = importlib.import_module(modname)
            except ImportError:
                continue
            self.check_init_docstring(mod, errors)
        if errors:
            msg = ''
            for (mod, value, init_doc) in errors:
                msg += '{}.{} has __init__.__doc__:\n{}\n\n'.format(mod.__name__, value, init_doc)
            self.fail(msg)
testing.run_module(__name__, __file__)