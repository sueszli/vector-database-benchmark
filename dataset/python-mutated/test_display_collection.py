from __future__ import annotations
import pytest
from ansible.cli.galaxy import _display_collection
from ansible.galaxy.dependency_resolution.dataclasses import Requirement

@pytest.fixture
def collection_object():
    if False:
        return 10

    def _cobj(fqcn='sandwiches.ham'):
        if False:
            i = 10
            return i + 15
        return Requirement(fqcn, '1.5.0', None, 'galaxy', None)
    return _cobj

def test_display_collection(capsys, collection_object):
    if False:
        return 10
    _display_collection(collection_object())
    (out, err) = capsys.readouterr()
    assert out == 'sandwiches.ham 1.5.0  \n'

def test_display_collections_small_max_widths(capsys, collection_object):
    if False:
        i = 10
        return i + 15
    _display_collection(collection_object(), 1, 1)
    (out, err) = capsys.readouterr()
    assert out == 'sandwiches.ham 1.5.0  \n'

def test_display_collections_large_max_widths(capsys, collection_object):
    if False:
        print('Hello World!')
    _display_collection(collection_object(), 20, 20)
    (out, err) = capsys.readouterr()
    assert out == 'sandwiches.ham       1.5.0               \n'

def test_display_collection_small_minimum_widths(capsys, collection_object):
    if False:
        for i in range(10):
            print('nop')
    _display_collection(collection_object('a.b'), min_cwidth=0, min_vwidth=0)
    (out, err) = capsys.readouterr()
    assert out == 'a.b        1.5.0  \n'