"""Unit tests for MultipleIncorrectSubmissions.py"""
from __future__ import annotations
from core.tests import test_utils
from extensions.issues.MultipleIncorrectSubmissions import MultipleIncorrectSubmissions

class MultipleIncorrectSubmissionsTests(test_utils.GenericTestBase):

    def test_trivial(self) -> None:
        if False:
            i = 10
            return i + 15
        pass