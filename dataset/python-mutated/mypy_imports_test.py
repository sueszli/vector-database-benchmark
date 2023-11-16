"""Unit tests for mypy_imports.py"""
from __future__ import annotations
from core.tests import test_utils
import mypy_imports

class MyPyImportsTests(test_utils.GenericTestBase):

    def test_trivial(self) -> None:
        if False:
            i = 10
            return i + 15
        pass