from __future__ import annotations
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from .clear import Clear

class ClearExt:
    """Extension for [Clear][rerun.archetypes.Clear]."""

    def __init__(self: Any, *, recursive: bool) -> None:
        if False:
            return 10
        '\n        Create a new instance of the Clear archetype.\n\n        Parameters\n        ----------\n        recursive:\n             Whether to recursively clear all children.\n        '
        self.__attrs_init__(is_recursive=recursive)

    @staticmethod
    def flat() -> Clear:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a non-recursive clear archetype.\n\n        This will empty all components of the associated entity at the logged timepoint.\n        Children will be left untouched.\n        '
        from .clear import Clear
        return Clear(recursive=False)

    @staticmethod
    def recursive() -> Clear:
        if False:
            while True:
                i = 10
        '\n        Returns a recursive clear archetype.\n\n        This will empty all components of the associated entity at the logged timepoint, as well as\n        all components of all its recursive children.\n        '
        from .clear import Clear
        return Clear(recursive=True)