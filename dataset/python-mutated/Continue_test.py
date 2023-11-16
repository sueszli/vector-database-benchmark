"""Unit tests for Continue.py"""
from __future__ import annotations
from core.tests import test_utils
from extensions.interactions.Continue import Continue

class ContinueTests(test_utils.GenericTestBase):

    def test_trivial(self) -> None:
        if False:
            print('Hello World!')
        pass