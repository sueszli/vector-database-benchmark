from sympy.external import import_module
from sympy.testing.pytest import warns

def test_no_stdlib_collections():
    if False:
        print('Hello World!')
    '\n    make sure we get the right collections when it is not part of a\n    larger list\n    '
    import collections
    matplotlib = import_module('matplotlib', import_kwargs={'fromlist': ['cm', 'collections']}, min_module_version='1.1.0', catch=(RuntimeError,))
    if matplotlib:
        assert collections != matplotlib.collections

def test_no_stdlib_collections2():
    if False:
        i = 10
        return i + 15
    '\n    make sure we get the right collections when it is not part of a\n    larger list\n    '
    import collections
    matplotlib = import_module('matplotlib', import_kwargs={'fromlist': ['collections']}, min_module_version='1.1.0', catch=(RuntimeError,))
    if matplotlib:
        assert collections != matplotlib.collections

def test_no_stdlib_collections3():
    if False:
        while True:
            i = 10
    'make sure we get the right collections with no catch'
    import collections
    matplotlib = import_module('matplotlib', import_kwargs={'fromlist': ['cm', 'collections']}, min_module_version='1.1.0')
    if matplotlib:
        assert collections != matplotlib.collections

def test_min_module_version_python3_basestring_error():
    if False:
        return 10
    with warns(UserWarning):
        import_module('mpmath', min_module_version='1000.0.1')