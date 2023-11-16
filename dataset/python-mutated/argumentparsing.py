"""This module contains helper functions related to parsing arguments for classes and methods.

Warning:
    Contents of this module are intended to be used internally by the library and *not* by the
    user. Changes to this module are not considered breaking changes and may not be documented in
    the changelog.
"""
from typing import Optional, Sequence, Tuple, TypeVar
T = TypeVar('T')

def parse_sequence_arg(arg: Optional[Sequence[T]]) -> Tuple[T, ...]:
    if False:
        for i in range(10):
            print('nop')
    'Parses an optional sequence into a tuple\n\n    Args:\n        arg (:obj:`Sequence`): The sequence to parse.\n\n    Returns:\n        :obj:`Tuple`: The sequence converted to a tuple or an empty tuple.\n    '
    return tuple(arg) if arg else ()