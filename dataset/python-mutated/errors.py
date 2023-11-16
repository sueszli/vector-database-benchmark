"""This module contains the exceptions used by the base_images module.
"""
from typing import Union
import dagger

class SanityCheckError(Exception):
    """Raised when a sanity check fails."""

    def __init__(self, error: Union[str, dagger.ExecError], *args: object) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(error, *args)