from __future__ import annotations
import pytest
from ansible.module_utils.common.parameters import _get_unsupported_parameters

@pytest.fixture
def argument_spec():
    if False:
        print('Hello World!')
    return {'state': {'aliases': ['status']}, 'enabled': {}}

@pytest.mark.parametrize(('module_parameters', 'legal_inputs', 'expected'), (({'fish': 'food'}, ['state', 'enabled'], set(['fish'])), ({'state': 'enabled', 'path': '/var/lib/path'}, None, set(['path'])), ({'state': 'enabled', 'path': '/var/lib/path'}, ['state', 'path'], set()), ({'state': 'enabled', 'path': '/var/lib/path'}, ['state'], set(['path'])), ({}, None, set()), ({'state': 'enabled'}, None, set()), ({'status': 'enabled', 'enabled': True, 'path': '/var/lib/path'}, None, set(['path'])), ({'status': 'enabled', 'enabled': True}, None, set())))
def test_check_arguments(argument_spec, module_parameters, legal_inputs, expected, mocker):
    if False:
        for i in range(10):
            print('nop')
    result = _get_unsupported_parameters(argument_spec, module_parameters, legal_inputs)
    assert result == expected