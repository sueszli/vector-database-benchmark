from __future__ import annotations
import pytest
from ansible import context
from ansible.cli.galaxy import GalaxyCLI

def test_execute_list_role_called(mocker):
    if False:
        for i in range(10):
            print('nop')
    'Make sure the correct method is called for a role'
    gc = GalaxyCLI(['ansible-galaxy', 'role', 'list'])
    context.CLIARGS._store = {'type': 'role'}
    execute_list_role_mock = mocker.patch('ansible.cli.galaxy.GalaxyCLI.execute_list_role', side_effect=AttributeError('raised intentionally'))
    execute_list_collection_mock = mocker.patch('ansible.cli.galaxy.GalaxyCLI.execute_list_collection', side_effect=AttributeError('raised intentionally'))
    with pytest.raises(AttributeError):
        gc.execute_list()
    assert execute_list_role_mock.call_count == 1
    assert execute_list_collection_mock.call_count == 0

def test_execute_list_collection_called(mocker):
    if False:
        while True:
            i = 10
    'Make sure the correct method is called for a collection'
    gc = GalaxyCLI(['ansible-galaxy', 'collection', 'list'])
    context.CLIARGS._store = {'type': 'collection'}
    execute_list_role_mock = mocker.patch('ansible.cli.galaxy.GalaxyCLI.execute_list_role', side_effect=AttributeError('raised intentionally'))
    execute_list_collection_mock = mocker.patch('ansible.cli.galaxy.GalaxyCLI.execute_list_collection', side_effect=AttributeError('raised intentionally'))
    with pytest.raises(AttributeError):
        gc.execute_list()
    assert execute_list_role_mock.call_count == 0
    assert execute_list_collection_mock.call_count == 1