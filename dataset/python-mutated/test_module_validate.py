from __future__ import annotations
from ansible.module_utils.common import warnings
from ansible.module_utils.common.arg_spec import ModuleArgumentSpecValidator, ValidationResult

def test_module_validate():
    if False:
        for i in range(10):
            print('nop')
    arg_spec = {'name': {}}
    parameters = {'name': 'larry'}
    expected = {'name': 'larry'}
    v = ModuleArgumentSpecValidator(arg_spec)
    result = v.validate(parameters)
    assert isinstance(result, ValidationResult)
    assert result.error_messages == []
    assert result._deprecations == []
    assert result._warnings == []
    assert result.validated_parameters == expected

def test_module_alias_deprecations_warnings(monkeypatch):
    if False:
        return 10
    monkeypatch.setattr(warnings, '_global_deprecations', [])
    arg_spec = {'path': {'aliases': ['source', 'src', 'flamethrower'], 'deprecated_aliases': [{'name': 'flamethrower', 'date': '2020-03-04'}]}}
    parameters = {'flamethrower': '/tmp', 'source': '/tmp'}
    expected = {'path': '/tmp', 'flamethrower': '/tmp', 'source': '/tmp'}
    v = ModuleArgumentSpecValidator(arg_spec)
    result = v.validate(parameters)
    assert result.validated_parameters == expected
    assert result._deprecations == [{'collection_name': None, 'date': '2020-03-04', 'msg': "Alias 'flamethrower' is deprecated. See the module docs for more information", 'version': None}]
    assert "Alias 'flamethrower' is deprecated" in warnings._global_deprecations[0]['msg']
    assert result._warnings == [{'alias': 'flamethrower', 'option': 'path'}]
    assert 'Both option path and its alias flamethrower are set' in warnings._global_warnings[0]