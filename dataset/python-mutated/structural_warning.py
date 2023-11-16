import warnings
from ludwig.utils.logging_utils import log_once

def warn_structure_refactor(old_module: str, new_module: str, direct: bool=True) -> None:
    if False:
        return 10
    'Create structure refactor warning to indicate modules new location post.\n\n    Only creates a warning once per module.\n    '
    old_module = old_module.replace('.py', '')
    if log_once(old_module):
        warning = f'The module `{old_module}` has been moved to `{new_module}` and the old location will be deprecated soon. Please adjust your imports to point to the new location.'
        if direct:
            warning += f' Example: Do a global search and replace `{old_module}` with `{new_module}`.'
        else:
            warning += f'\nATTENTION: This module may have been split or refactored. Please check the contents of `{new_module}` before making changes.'
        with warnings.catch_warnings():
            warnings.simplefilter('always')
            warnings.warn(warning, DeprecationWarning, stacklevel=3)