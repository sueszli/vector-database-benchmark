from __future__ import annotations
import pytest
from ansible.cli.galaxy import _get_collection_widths
from ansible.galaxy.dependency_resolution.dataclasses import Requirement

@pytest.fixture
def collection_objects():
    if False:
        return 10
    collection_ham = Requirement('sandwiches.ham', '1.5.0', None, 'galaxy', None)
    collection_pbj = Requirement('sandwiches.pbj', '2.5', None, 'galaxy', None)
    collection_reuben = Requirement('sandwiches.reuben', '4', None, 'galaxy', None)
    return [collection_ham, collection_pbj, collection_reuben]

def test_get_collection_widths(collection_objects):
    if False:
        while True:
            i = 10
    assert _get_collection_widths(collection_objects) == (17, 5)

def test_get_collection_widths_single_collection(mocker):
    if False:
        while True:
            i = 10
    mocked_collection = Requirement('sandwiches.club', '3.0.0', None, 'galaxy', None)
    mocker.patch('ansible.cli.galaxy.is_iterable', return_value=False)
    assert _get_collection_widths(mocked_collection) == (15, 5)