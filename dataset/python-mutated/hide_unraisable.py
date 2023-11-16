"""Temporary plugin to prevent stdout noise pollution from finalization of abandoned generators under Python 3.12"""
from __future__ import annotations
import sys
import typing as t
if t.TYPE_CHECKING:
    from pylint.lint import PyLinter

def _mask_finalizer_valueerror(ur: t.Any) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Mask only ValueErrors from finalizing abandoned generators; delegate everything else'
    if ur.exc_type is ValueError and 'generator already executing' in str(ur.exc_value):
        return
    sys.__unraisablehook__(ur)

def register(linter: PyLinter) -> None:
    if False:
        for i in range(10):
            print('nop')
    'PyLint plugin registration entrypoint'
    if sys.version_info >= (3, 12):
        sys.unraisablehook = _mask_finalizer_valueerror