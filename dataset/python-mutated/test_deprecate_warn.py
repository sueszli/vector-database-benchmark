from __future__ import annotations
import json
import pytest
from ansible.module_utils.common import warnings

@pytest.mark.parametrize('stdin', [{}], indirect=['stdin'])
def test_warn(am, capfd):
    if False:
        for i in range(10):
            print('nop')
    am.warn('warning1')
    with pytest.raises(SystemExit):
        am.exit_json(warnings=['warning2'])
    (out, err) = capfd.readouterr()
    assert json.loads(out)['warnings'] == ['warning1', 'warning2']

@pytest.mark.parametrize('stdin', [{}], indirect=['stdin'])
def test_deprecate(am, capfd, monkeypatch):
    if False:
        i = 10
        return i + 15
    monkeypatch.setattr(warnings, '_global_deprecations', [])
    am.deprecate('deprecation1')
    am.deprecate('deprecation2', '2.3')
    am.deprecate('deprecation3', version='2.4')
    am.deprecate('deprecation4', date='2020-03-10')
    am.deprecate('deprecation5', collection_name='ansible.builtin')
    am.deprecate('deprecation6', '2.3', collection_name='ansible.builtin')
    am.deprecate('deprecation7', version='2.4', collection_name='ansible.builtin')
    am.deprecate('deprecation8', date='2020-03-10', collection_name='ansible.builtin')
    with pytest.raises(SystemExit):
        am.exit_json(deprecations=['deprecation9', ('deprecation10', '2.4')])
    (out, err) = capfd.readouterr()
    output = json.loads(out)
    assert 'warnings' not in output or output['warnings'] == []
    assert output['deprecations'] == [{u'msg': u'deprecation1', u'version': None, u'collection_name': None}, {u'msg': u'deprecation2', u'version': '2.3', u'collection_name': None}, {u'msg': u'deprecation3', u'version': '2.4', u'collection_name': None}, {u'msg': u'deprecation4', u'date': '2020-03-10', u'collection_name': None}, {u'msg': u'deprecation5', u'version': None, u'collection_name': 'ansible.builtin'}, {u'msg': u'deprecation6', u'version': '2.3', u'collection_name': 'ansible.builtin'}, {u'msg': u'deprecation7', u'version': '2.4', u'collection_name': 'ansible.builtin'}, {u'msg': u'deprecation8', u'date': '2020-03-10', u'collection_name': 'ansible.builtin'}, {u'msg': u'deprecation9', u'version': None, u'collection_name': None}, {u'msg': u'deprecation10', u'version': '2.4', u'collection_name': None}]

@pytest.mark.parametrize('stdin', [{}], indirect=['stdin'])
def test_deprecate_without_list(am, capfd):
    if False:
        print('Hello World!')
    with pytest.raises(SystemExit):
        am.exit_json(deprecations='Simple deprecation warning')
    (out, err) = capfd.readouterr()
    output = json.loads(out)
    assert 'warnings' not in output or output['warnings'] == []
    assert output['deprecations'] == [{u'msg': u'Simple deprecation warning', u'version': None, u'collection_name': None}]

@pytest.mark.parametrize('stdin', [{}], indirect=['stdin'])
def test_deprecate_without_list_version_date_not_set(am, capfd):
    if False:
        print('Hello World!')
    with pytest.raises(AssertionError) as ctx:
        am.deprecate('Simple deprecation warning', date='', version='')
    assert ctx.value.args[0] == 'implementation error -- version and date must not both be set'