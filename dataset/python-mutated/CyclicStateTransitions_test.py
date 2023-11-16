"""Unit tests for CyclicStateTransitions.py"""
from __future__ import annotations
from core.tests import test_utils
from extensions.issues.CyclicStateTransitions import CyclicStateTransitions

class CyclicStateTransitionsTests(test_utils.GenericTestBase):

    def test_trivial(self) -> None:
        if False:
            i = 10
            return i + 15
        pass