"""Tests for methods in the issue registry."""
from __future__ import annotations
from core.domain import playthrough_issue_registry
from core.tests import test_utils
from extensions.issues.CyclicStateTransitions import CyclicStateTransitions
from extensions.issues.EarlyQuit import EarlyQuit
from extensions.issues.MultipleIncorrectSubmissions import MultipleIncorrectSubmissions

class IssueRegistryUnitTests(test_utils.GenericTestBase):
    """Test for the issue registry."""

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        self.issues_dict = {'EarlyQuit': EarlyQuit.EarlyQuit, 'CyclicStateTransitions': CyclicStateTransitions.CyclicStateTransitions, 'MultipleIncorrectSubmissions': MultipleIncorrectSubmissions.MultipleIncorrectSubmissions}
        self.invalid_issue_type = 'InvalidIssueType'

    def tearDown(self) -> None:
        if False:
            print('Hello World!')
        playthrough_issue_registry.Registry._issues = {}
        super().tearDown()

    def test_issue_registry(self) -> None:
        if False:
            return 10
        'Do some sanity checks on the issue registry.'
        self.assertEqual(len(playthrough_issue_registry.Registry.get_all_issues()), 3)

    def test_correct_issue_registry_types(self) -> None:
        if False:
            return 10
        'Tests issue registry for fetching of issue instances of correct\n        issue types.\n        '
        for (issue_type, _class) in self.issues_dict.items():
            self.assertIsInstance(playthrough_issue_registry.Registry.get_issue_by_type(issue_type), _class)

    def test_incorrect_issue_registry_types(self) -> None:
        if False:
            while True:
                i = 10
        'Tests that an error is raised when fetching an incorrect issue\n        type.\n        '
        with self.assertRaisesRegex(KeyError, self.invalid_issue_type):
            playthrough_issue_registry.Registry.get_issue_by_type(self.invalid_issue_type)