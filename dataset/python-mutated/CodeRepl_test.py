"""Unit tests for CodeRepl.py"""
from __future__ import annotations
from core.tests import test_utils
from extensions.interactions.CodeRepl import CodeRepl

class CodeReplTests(test_utils.GenericTestBase):

    def test_trivial(self) -> None:
        if False:
            return 10
        pass