"""Tests for methods in the action registry."""
from __future__ import annotations
from core.domain import action_registry
from core.tests import test_utils

class ActionRegistryUnitTests(test_utils.GenericTestBase):
    """Test for the action registry."""

    def test_action_registry(self) -> None:
        if False:
            while True:
                i = 10
        'Do some sanity checks on the action registry.'
        self.assertEqual(len(action_registry.Registry.get_all_actions()), 3)

    def test_cannot_get_action_by_invalid_type(self) -> None:
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(KeyError, 'fakeAction'):
            action_registry.Registry.get_action_by_type('fakeAction')

    def test_can_get_action_by_valid_type(self) -> None:
        if False:
            i = 10
            return i + 15
        self.assertIsNotNone(action_registry.Registry.get_action_by_type('ExplorationStart'))