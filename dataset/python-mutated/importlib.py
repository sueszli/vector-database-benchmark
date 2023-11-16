from __future__ import annotations
from ansible.module_utils.common.warnings import deprecate

def __getattr__(importable_name):
    if False:
        return 10
    'Inject import-time deprecation warnings.\n\n    Specifically, for ``import_module()``.\n    '
    if importable_name == 'import_module':
        deprecate(msg=f'The `ansible.module_utils.compat.importlib.{importable_name}` function is deprecated.', version='2.19')
        from importlib import import_module
        return import_module
    raise AttributeError(f'cannot import name {importable_name!r} has no attribute ({__file__!s})')