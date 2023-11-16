"""Unit tests for EndExploration.py"""
from __future__ import annotations
from core.tests import test_utils
from extensions.interactions.EndExploration import EndExploration

class EndExplorationTests(test_utils.GenericTestBase):

    def test_trivial(self) -> None:
        if False:
            i = 10
            return i + 15
        pass