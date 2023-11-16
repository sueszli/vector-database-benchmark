"""This module contains functionality used for transition warnings issued by this library.

It was created to prevent circular imports that would be caused by creating the warnings
inside warnings.py.

.. versionadded:: 20.2
"""
from typing import Any, Callable, Type
from telegram._utils.warnings import warn
from telegram.warnings import PTBDeprecationWarning

def warn_about_deprecated_arg_return_new_arg(deprecated_arg: Any, new_arg: Any, deprecated_arg_name: str, new_arg_name: str, bot_api_version: str, stacklevel: int=2, warn_callback: Callable[[str, Type[Warning], int], None]=warn) -> Any:
    if False:
        return 10
    'A helper function for the transition in API when argument is renamed.\n\n    Checks the `deprecated_arg` and `new_arg` objects; warns if non-None `deprecated_arg` object\n    was passed. Returns `new_arg` object (either the one originally passed by the user or the one\n    that user passed as `deprecated_arg`).\n\n    Raises `ValueError` if both `deprecated_arg` and `new_arg` objects were passed, and they are\n    different.\n    '
    if deprecated_arg and new_arg and (deprecated_arg != new_arg):
        raise ValueError(f"You passed different entities as '{deprecated_arg_name}' and '{new_arg_name}'. The parameter '{deprecated_arg_name}' was renamed to '{new_arg_name}' in Bot API {bot_api_version}. We recommend using '{new_arg_name}' instead of '{deprecated_arg_name}'.")
    if deprecated_arg:
        warn_callback(f"Bot API {bot_api_version} renamed the argument '{deprecated_arg_name}' to '{new_arg_name}'.", PTBDeprecationWarning, stacklevel + 1)
        return deprecated_arg
    return new_arg

def warn_about_deprecated_attr_in_property(deprecated_attr_name: str, new_attr_name: str, bot_api_version: str, stacklevel: int=2) -> None:
    if False:
        while True:
            i = 10
    'A helper function for the transition in API when attribute is renamed. Call from properties.\n\n    The properties replace deprecated attributes in classes and issue these deprecation warnings.\n    '
    warn(f"Bot API {bot_api_version} renamed the attribute '{deprecated_attr_name}' to '{new_attr_name}'.", PTBDeprecationWarning, stacklevel=stacklevel + 1)