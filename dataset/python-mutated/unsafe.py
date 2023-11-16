from typing import TypeVar
from returns.io import IO
_ValueType = TypeVar('_ValueType')

def unsafe_perform_io(wrapped_in_io: IO[_ValueType]) -> _ValueType:
    if False:
        while True:
            i = 10
    '\n    Compatibility utility and escape mechanism from ``IO`` world.\n\n    Just unwraps the internal value\n    from :class:`returns.io.IO` container.\n    Should be used with caution!\n    Since it might be overused by lazy and ignorant developers.\n\n    It is recommended to have only one place (module / file)\n    in your program where you allow unsafe operations.\n\n    We recommend to use ``import-linter`` to enforce this rule.\n\n    .. code:: python\n\n      >>> from returns.io import IO\n      >>> assert unsafe_perform_io(IO(1)) == 1\n\n    See also:\n        - https://github.com/seddonym/import-linter\n\n    '
    return wrapped_in_io._inner_value