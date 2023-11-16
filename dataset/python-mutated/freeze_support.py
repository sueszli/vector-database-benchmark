"""Provides a function to report all internal modules for using freezing
tools."""
import types
from typing import Iterator
from typing import List
from typing import Union

def freeze_includes() -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Return a list of module names used by pytest that should be\n    included by cx_freeze.'
    import _pytest
    result = list(_iter_all_modules(_pytest))
    return result

def _iter_all_modules(package: Union[str, types.ModuleType], prefix: str='') -> Iterator[str]:
    if False:
        for i in range(10):
            print('nop')
    "Iterate over the names of all modules that can be found in the given\n    package, recursively.\n\n        >>> import _pytest\n        >>> list(_iter_all_modules(_pytest))\n        ['_pytest._argcomplete', '_pytest._code.code', ...]\n    "
    import os
    import pkgutil
    if isinstance(package, str):
        path = package
    else:
        package_path = package.__path__
        (path, prefix) = (package_path[0], package.__name__ + '.')
    for (_, name, is_package) in pkgutil.iter_modules([path]):
        if is_package:
            for m in _iter_all_modules(os.path.join(path, name), prefix=name + '.'):
                yield (prefix + m)
        else:
            yield (prefix + name)