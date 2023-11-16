"""Unit tests for PencilCodeEditor.py"""
from __future__ import annotations
from core.tests import test_utils
from extensions.interactions.PencilCodeEditor import PencilCodeEditor

class PencilCodeEditorTests(test_utils.GenericTestBase):

    def test_trivial(self) -> None:
        if False:
            while True:
                i = 10
        pass