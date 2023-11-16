from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import sys
import six
from turicreate import __version__
USE_MINIMAL = False

def is_minimal_pkg():
    if False:
        while True:
            i = 10
    return USE_MINIMAL
if six.PY2:
    import imp
else:
    import importlib
if six.PY2:

    class _ImportLockContext:

        def __enter__(self):
            if False:
                while True:
                    i = 10
            imp.acquire_lock()

        def __exit__(self, exc_type, exc_value, exc_traceback):
            if False:
                while True:
                    i = 10
            imp.release_lock()
else:
    from importlib._bootstrap import _ImportLockContext

def _minimal_package_import_check(name):
    if False:
        return 10
    '\n    only support `import ...`, no `from ... import` or `import ... as`\n    '
    with _ImportLockContext():
        _ret = sys.modules.get(name, None)
    if _ret is not None:
        return _ret
    try:
        if six.PY2:
            if not all([x.isalnum() or x in '._' for x in name]):
                raise ValueError('invalid module name')
            exec('import %s as __mpkg_ret' % name)
            return __mpkg_ret
        else:
            return importlib.import_module(name)
    except ImportError as e:
        if USE_MINIMAL:
            if '.' in name:
                name = name.split('.')[0]
            pos = __version__.rfind('+')
            if pos != -1:
                version = __version__[:pos]
            else:
                version = __version__
            emsg = str(e)
            emsg = '{}.\nThis is a minimal package for SFrame only, without {} pinned as a dependency. You can try install all required packages by installing the full package. For example:\npip install --force-reinstall turicreate=={}\n'.format(emsg, name, version)
            if six.PY2:
                args = list(e.args)
                if args:
                    args[0] = emsg
                else:
                    args = (emsg,)
                e.args = tuple(args)
                e.message = emsg
            else:
                e.msg = emsg
        raise e