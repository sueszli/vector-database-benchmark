"""Import mangling.
See mangling.md for details.
"""
import re
_mangle_index = 0

class PackageMangler:
    """
    Used on import, to ensure that all modules imported have a shared mangle parent.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        global _mangle_index
        self._mangle_index = _mangle_index
        _mangle_index += 1
        self._mangle_parent = f'<torch_package_{self._mangle_index}>'

    def mangle(self, name) -> str:
        if False:
            print('Hello World!')
        assert len(name) != 0
        return self._mangle_parent + '.' + name

    def demangle(self, mangled: str) -> str:
        if False:
            while True:
                i = 10
        '\n        Note: This only demangles names that were mangled by this specific\n        PackageMangler. It will pass through names created by a different\n        PackageMangler instance.\n        '
        if mangled.startswith(self._mangle_parent + '.'):
            return mangled.partition('.')[2]
        return mangled

    def parent_name(self):
        if False:
            for i in range(10):
                print('nop')
        return self._mangle_parent

def is_mangled(name: str) -> bool:
    if False:
        while True:
            i = 10
    return bool(re.match('<torch_package_\\d+>', name))

def demangle(name: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Note: Unlike PackageMangler.demangle, this version works on any\n    mangled name, irrespective of which PackageMangler created it.\n    '
    if is_mangled(name):
        (first, sep, last) = name.partition('.')
        return last if len(sep) != 0 else ''
    return name

def get_mangle_prefix(name: str) -> str:
    if False:
        print('Hello World!')
    return name.partition('.')[0] if is_mangled(name) else name