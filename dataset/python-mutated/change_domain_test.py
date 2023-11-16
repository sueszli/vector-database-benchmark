"""Unit tests for change_domain.py"""
from __future__ import annotations
from core import feconf
from core.domain import change_domain
from core.tests import test_utils

class ChangeDomainTests(test_utils.GenericTestBase):

    def test_that_domain_object_is_created_correctly(self) -> None:
        if False:
            i = 10
            return i + 15
        change_object = change_domain.BaseChange({'cmd': feconf.CMD_DELETE_COMMIT})
        expected_change_object_dict = {'cmd': feconf.CMD_DELETE_COMMIT}
        self.assertEqual(change_object.to_dict(), expected_change_object_dict)