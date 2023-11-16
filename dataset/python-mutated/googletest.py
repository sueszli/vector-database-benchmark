"""Imports absltest as a replacement for testing.pybase.googletest."""
import atexit
import os
import sys
import tempfile
from absl import app
from absl.testing.absltest import *
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export
Benchmark = benchmark.TensorFlowBenchmark
absltest_main = main
_googletest_temp_dir = ''

def g_main(argv):
    if False:
        for i in range(10):
            print('nop')
    'Delegate to absltest.main.'
    absltest_main(argv=argv)

def main(argv=None):
    if False:
        while True:
            i = 10

    def main_wrapper():
        if False:
            print('Hello World!')
        args = argv
        if args is None:
            args = sys.argv
        return app.run(main=g_main, argv=args)
    benchmark.benchmarks_main(true_main=main_wrapper, argv=argv)

def GetTempDir():
    if False:
        for i in range(10):
            print('nop')
    'Return a temporary directory for tests to use.'
    global _googletest_temp_dir
    if not _googletest_temp_dir:
        if os.environ.get('TEST_TMPDIR'):
            temp_dir = tempfile.mkdtemp(prefix=os.environ['TEST_TMPDIR'])
        else:
            first_frame = tf_inspect.stack()[-1][0]
            temp_dir = os.path.join(tempfile.gettempdir(), os.path.basename(tf_inspect.getfile(first_frame)))
            temp_dir = tempfile.mkdtemp(prefix=temp_dir.rstrip('.py'))
        temp_dir = temp_dir.replace('/', os.sep)

        def delete_temp_dir(dirname=temp_dir):
            if False:
                i = 10
                return i + 15
            try:
                file_io.delete_recursively(dirname)
            except errors.OpError as e:
                logging.error('Error removing %s: %s', dirname, e)
        atexit.register(delete_temp_dir)
        _googletest_temp_dir = temp_dir
    return _googletest_temp_dir

def test_src_dir_path(relative_path):
    if False:
        print('Hello World!')
    'Creates an absolute test srcdir path given a relative path.\n\n  Args:\n    relative_path: a path relative to tensorflow root.\n      e.g. "contrib/session_bundle/example".\n\n  Returns:\n    An absolute path to the linked in runfiles.\n  '
    return os.path.join(os.environ['TEST_SRCDIR'], 'org_tensorflow/tensorflow', relative_path)

def StatefulSessionAvailable():
    if False:
        while True:
            i = 10
    return False

@tf_export(v1=['test.StubOutForTesting'])
class StubOutForTesting(object):
    """Support class for stubbing methods out for unit testing.

  Sample Usage:

  You want os.path.exists() to always return true during testing.

     stubs = StubOutForTesting()
     stubs.Set(os.path, 'exists', lambda x: 1)
       ...
     stubs.CleanUp()

  The above changes os.path.exists into a lambda that returns 1.  Once
  the ... part of the code finishes, the CleanUp() looks up the old
  value of os.path.exists and restores it.
  """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.cache = []
        self.stubs = []

    def __del__(self):
        if False:
            print('Hello World!')
        'Do not rely on the destructor to undo your stubs.\n\n    You cannot guarantee exactly when the destructor will get called without\n    relying on implementation details of a Python VM that may change.\n    '
        self.CleanUp()

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        return self

    def __exit__(self, unused_exc_type, unused_exc_value, unused_tb):
        if False:
            for i in range(10):
                print('nop')
        self.CleanUp()

    def CleanUp(self):
        if False:
            for i in range(10):
                print('nop')
        'Undoes all SmartSet() & Set() calls, restoring original definitions.'
        self.SmartUnsetAll()
        self.UnsetAll()

    def SmartSet(self, obj, attr_name, new_attr):
        if False:
            while True:
                i = 10
        'Replace obj.attr_name with new_attr.\n\n    This method is smart and works at the module, class, and instance level\n    while preserving proper inheritance. It will not stub out C types however\n    unless that has been explicitly allowed by the type.\n\n    This method supports the case where attr_name is a staticmethod or a\n    classmethod of obj.\n\n    Notes:\n      - If obj is an instance, then it is its class that will actually be\n        stubbed. Note that the method Set() does not do that: if obj is\n        an instance, it (and not its class) will be stubbed.\n      - The stubbing is using the builtin getattr and setattr. So, the __get__\n        and __set__ will be called when stubbing (TODO: A better idea would\n        probably be to manipulate obj.__dict__ instead of getattr() and\n        setattr()).\n\n    Args:\n      obj: The object whose attributes we want to modify.\n      attr_name: The name of the attribute to modify.\n      new_attr: The new value for the attribute.\n\n    Raises:\n      AttributeError: If the attribute cannot be found.\n    '
        (_, obj) = tf_decorator.unwrap(obj)
        if tf_inspect.ismodule(obj) or (not tf_inspect.isclass(obj) and attr_name in obj.__dict__):
            orig_obj = obj
            orig_attr = getattr(obj, attr_name)
        else:
            if not tf_inspect.isclass(obj):
                mro = list(tf_inspect.getmro(obj.__class__))
            else:
                mro = list(tf_inspect.getmro(obj))
            mro.reverse()
            orig_attr = None
            found_attr = False
            for cls in mro:
                try:
                    orig_obj = cls
                    orig_attr = getattr(obj, attr_name)
                    found_attr = True
                except AttributeError:
                    continue
            if not found_attr:
                raise AttributeError('Attribute not found.')
        old_attribute = obj.__dict__.get(attr_name)
        if old_attribute is not None and isinstance(old_attribute, staticmethod):
            orig_attr = staticmethod(orig_attr)
        self.stubs.append((orig_obj, attr_name, orig_attr))
        setattr(orig_obj, attr_name, new_attr)

    def SmartUnsetAll(self):
        if False:
            i = 10
            return i + 15
        'Reverses SmartSet() calls, restoring things to original definitions.\n\n    This method is automatically called when the StubOutForTesting()\n    object is deleted; there is no need to call it explicitly.\n\n    It is okay to call SmartUnsetAll() repeatedly, as later calls have\n    no effect if no SmartSet() calls have been made.\n    '
        for args in reversed(self.stubs):
            setattr(*args)
        self.stubs = []

    def Set(self, parent, child_name, new_child):
        if False:
            i = 10
            return i + 15
        "In parent, replace child_name's old definition with new_child.\n\n    The parent could be a module when the child is a function at\n    module scope.  Or the parent could be a class when a class' method\n    is being replaced.  The named child is set to new_child, while the\n    prior definition is saved away for later, when UnsetAll() is\n    called.\n\n    This method supports the case where child_name is a staticmethod or a\n    classmethod of parent.\n\n    Args:\n      parent: The context in which the attribute child_name is to be changed.\n      child_name: The name of the attribute to change.\n      new_child: The new value of the attribute.\n    "
        old_child = getattr(parent, child_name)
        old_attribute = parent.__dict__.get(child_name)
        if old_attribute is not None and isinstance(old_attribute, staticmethod):
            old_child = staticmethod(old_child)
        self.cache.append((parent, old_child, child_name))
        setattr(parent, child_name, new_child)

    def UnsetAll(self):
        if False:
            print('Hello World!')
        'Reverses Set() calls, restoring things to their original definitions.\n\n    This method is automatically called when the StubOutForTesting()\n    object is deleted; there is no need to call it explicitly.\n\n    It is okay to call UnsetAll() repeatedly, as later calls have no\n    effect if no Set() calls have been made.\n    '
        for (parent, old_child, child_name) in reversed(self.cache):
            setattr(parent, child_name, old_child)
        self.cache = []