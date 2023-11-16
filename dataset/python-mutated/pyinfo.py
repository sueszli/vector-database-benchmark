from __future__ import annotations
'Utilities to find the site and prefix information of a Python executable.\n\nThis file MUST remain compatible with all Python 3.8+ versions. Since we cannot make any\nassumptions about the Python being executed, this module should not use *any* dependencies outside\nof the standard library found in Python 3.8. This file is run each mypy run, so it should be kept\nas fast as possible.\n'
import sys
if __name__ == '__main__':
    if sys.version_info < (3, 11):
        old_sys_path = sys.path
        sys.path = sys.path[1:]
        import types
        sys.path = old_sys_path
import os
import site
import sysconfig

def getsitepackages() -> list[str]:
    if False:
        i = 10
        return i + 15
    res = []
    if hasattr(site, 'getsitepackages'):
        res.extend(site.getsitepackages())
        if hasattr(site, 'getusersitepackages') and site.ENABLE_USER_SITE:
            res.insert(0, site.getusersitepackages())
    else:
        res = [sysconfig.get_paths()['purelib']]
    return res

def getsyspath() -> list[str]:
    if False:
        for i in range(10):
            print('nop')
    stdlib_zip = os.path.join(sys.base_exec_prefix, getattr(sys, 'platlibdir', 'lib'), f'python{sys.version_info.major}{sys.version_info.minor}.zip')
    stdlib = sysconfig.get_path('stdlib')
    stdlib_ext = os.path.join(stdlib, 'lib-dynload')
    excludes = {stdlib_zip, stdlib, stdlib_ext}
    offset = 0 if sys.version_info >= (3, 11) and sys.flags.safe_path else 1
    abs_sys_path = (os.path.abspath(p) for p in sys.path[offset:])
    return [p for p in abs_sys_path if p not in excludes]

def getsearchdirs() -> tuple[list[str], list[str]]:
    if False:
        return 10
    return (getsyspath(), getsitepackages())
if __name__ == '__main__':
    if sys.argv[-1] == 'getsearchdirs':
        print(repr(getsearchdirs()))
    else:
        print('ERROR: incorrect argument to pyinfo.py.', file=sys.stderr)
        sys.exit(1)