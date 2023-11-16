"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.nftables
"""
import json
import pytest
import salt.modules.nftables as nftables
import salt.utils.files
from salt.exceptions import CommandExecutionError
from tests.support.mock import MagicMock, mock_open, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        for i in range(10):
            print('nop')
    return {nftables: {}}

def test_version():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it return version from nftables --version\n    '
    mock = MagicMock(return_value='nf_tables 0.3-1')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.version() == '0.3-1'

def test_build_rule():
    if False:
        return 10
    '\n    Test if it build a well-formatted nftables rule based on kwargs.\n    '
    assert nftables.build_rule(full='True') == {'result': False, 'rule': '', 'comment': 'Table needs to be specified'}
    assert nftables.build_rule(table='filter', full='True') == {'result': False, 'rule': '', 'comment': 'Chain needs to be specified'}
    assert nftables.build_rule(table='filter', chain='input', full='True') == {'result': False, 'rule': '', 'comment': 'Command needs to be specified'}
    assert nftables.build_rule(table='filter', chain='input', command='insert', position='3', full='True') == {'result': True, 'rule': 'nft insert rule ip filter input position 3 ', 'comment': 'Successfully built rule'}
    assert nftables.build_rule(table='filter', chain='input', command='insert', full='True') == {'result': True, 'rule': 'nft insert rule ip filter input ', 'comment': 'Successfully built rule'}
    assert nftables.build_rule(table='filter', chain='input', command='halt', full='True') == {'result': True, 'rule': 'nft halt rule ip filter input ', 'comment': 'Successfully built rule'}
    assert nftables.build_rule(table='filter', chain='input', command='insert', position='3', full='True', connstate='related,established', saddr='10.0.0.1', daddr='10.0.0.2', jump='accept') == {'result': True, 'rule': 'nft insert rule ip filter input position 3 ct state { related,established } ip saddr 10.0.0.1 ip daddr 10.0.0.2 accept', 'comment': 'Successfully built rule'}
    assert nftables.build_rule() == {'result': True, 'rule': '', 'comment': ''}

def test_get_saved_rules():
    if False:
        print('Hello World!')
    '\n    Test if it return a data structure of the rules in the conf file\n    '
    with patch.dict(nftables.__grains__, {'os_family': 'Debian'}):
        with patch.object(salt.utils.files, 'fopen', MagicMock(mock_open())):
            assert nftables.get_saved_rules() == []

def test_list_tables():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it return a data structure of the current, in-memory tables\n    '
    list_tables = [{'family': 'inet', 'name': 'filter', 'handle': 2}]
    list_tables_mock = MagicMock(return_value=list_tables)
    with patch.object(nftables, 'list_tables', list_tables_mock):
        assert nftables.list_tables() == list_tables
    list_tables_mock = MagicMock(return_value=[])
    with patch.object(nftables, 'list_tables', list_tables_mock):
        assert nftables.list_tables() == []

def test_get_rules():
    if False:
        print('Hello World!')
    '\n    Test if it return a data structure of the current, in-memory rules\n    '
    list_tables_mock = MagicMock(return_value=[{'family': 'inet', 'name': 'filter', 'handle': 2}])
    list_rules_return = 'table inet filter {\n        chain input {\n            type filter hook input priority 0; policy accept;\n        }\n\n        chain forward {\n            type filter hook forward priority 0; policy accept;\n        }\n\n        chain output {\n            type filter hook output priority 0; policy accept;\n        }\n    }'
    list_rules_mock = MagicMock(return_value=list_rules_return)
    expected = [list_rules_return]
    with patch.object(nftables, 'list_tables', list_tables_mock):
        with patch.dict(nftables.__salt__, {'cmd.run': list_rules_mock}):
            assert nftables.get_rules() == expected
    list_tables_mock = MagicMock(return_value=[])
    with patch.object(nftables, 'list_tables', list_tables_mock):
        assert nftables.get_rules() == []

def test_get_rules_json():
    if False:
        i = 10
        return i + 15
    '\n    Test if it return a data structure of the current, in-memory rules\n    '
    list_rules_return = '\n    {\n      "nftables": [\n        {\n          "table": {\n            "family": "ip",\n            "name": "filter",\n            "handle": 47\n          }\n        },\n        {\n          "chain": {\n            "family": "ip",\n            "table": "filter",\n            "name": "input",\n            "handle": 1,\n            "type": "filter",\n            "hook": "input",\n            "prio": 0,\n            "policy": "accept"\n          }\n        },\n        {\n          "chain": {\n            "family": "ip",\n            "table": "filter",\n            "name": "forward",\n            "handle": 2,\n            "type": "filter",\n            "hook": "forward",\n            "prio": 0,\n            "policy": "accept"\n          }\n        },\n        {\n          "chain": {\n            "family": "ip",\n            "table": "filter",\n            "name": "output",\n            "handle": 3,\n            "type": "filter",\n            "hook": "output",\n            "prio": 0,\n            "policy": "accept"\n          }\n        }\n      ]\n    }\n    '
    list_rules_mock = MagicMock(return_value=list_rules_return)
    expected = json.loads(list_rules_return)['nftables']
    with patch.dict(nftables.__salt__, {'cmd.run': list_rules_mock}):
        assert nftables.get_rules_json() == expected
    list_rules_mock = MagicMock(return_value=[])
    with patch.dict(nftables.__salt__, {'cmd.run': list_rules_mock}):
        assert nftables.get_rules_json() == []

def test_save():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it save the current in-memory rules to disk\n    '
    with patch.dict(nftables.__grains__, {'os_family': 'Debian'}):
        mock = MagicMock(return_value=False)
        with patch.dict(nftables.__salt__, {'file.directory_exists': mock}):
            with patch.dict(nftables.__salt__, {'cmd.run': mock}):
                with patch.object(salt.utils.files, 'fopen', MagicMock(mock_open())):
                    assert nftables.save() == '#! nft -f\n\n'
                with patch.object(salt.utils.files, 'fopen', MagicMock(side_effect=IOError)):
                    pytest.raises(CommandExecutionError, nftables.save)

def test_get_rule_handle():
    if False:
        return 10
    '\n    Test if it get the handle for a particular rule\n    '
    assert nftables.get_rule_handle() == {'result': False, 'comment': 'Chain needs to be specified'}
    assert nftables.get_rule_handle(chain='input') == {'result': False, 'comment': 'Rule needs to be specified'}
    _ru = 'input tcp dport 22 log accept'
    ret = {'result': False, 'comment': 'Table filter in family ipv4 does not exist'}
    mock = MagicMock(return_value='')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.get_rule_handle(chain='input', rule=_ru) == ret
    ret = {'result': False, 'comment': 'Chain input in table filter in family ipv4 does not exist'}
    mock = MagicMock(return_value='table ip filter')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.get_rule_handle(chain='input', rule=_ru) == ret
    ret = {'result': False, 'comment': 'Rule input tcp dport 22 log accept chain input in table filter in family ipv4 does not exist'}
    ret1 = {'result': False, 'comment': 'Could not find rule input tcp dport 22 log accept'}
    with patch.object(nftables, 'check_table', MagicMock(return_value={'result': True, 'comment': ''})):
        with patch.object(nftables, 'check_chain', MagicMock(return_value={'result': True, 'comment': ''})):
            _ret1 = {'result': False, 'comment': 'Rule input tcp dport 22 log accept chain input in table filter in family ipv4 does not exist'}
            _ret2 = {'result': True, 'comment': ''}
            with patch.object(nftables, 'check', MagicMock(side_effect=[_ret1, _ret2])):
                assert nftables.get_rule_handle(chain='input', rule=_ru) == ret
                _ru = 'input tcp dport 22 log accept'
                mock = MagicMock(return_value='')
                with patch.dict(nftables.__salt__, {'cmd.run': mock}):
                    assert nftables.get_rule_handle(chain='input', rule=_ru) == ret1

def test_check():
    if False:
        print('Hello World!')
    '\n    Test if it check for the existence of a rule in the table and chain\n    '
    assert nftables.check() == {'result': False, 'comment': 'Chain needs to be specified'}
    assert nftables.check(chain='input') == {'result': False, 'comment': 'Rule needs to be specified'}
    _ru = 'tcp dport 22 log accept'
    ret = {'result': False, 'comment': 'Table filter in family ipv4 does not exist'}
    mock = MagicMock(return_value='')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.check(chain='input', rule=_ru) == ret
    mock = MagicMock(return_value='table ip filter')
    ret = {'result': False, 'comment': 'Chain input in table filter in family ipv4 does not exist'}
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.check(chain='input', rule=_ru) == ret
    mock = MagicMock(return_value='table ip filter chain input {{')
    ret = {'result': False, 'comment': 'Rule tcp dport 22 log accept in chain input in table filter in family ipv4 does not exist'}
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.check(chain='input', rule=_ru) == ret
    r_val = 'table ip filter chain input {{ input tcp dport 22 log accept #'
    mock = MagicMock(return_value=r_val)
    ret = {'result': True, 'comment': 'Rule tcp dport 22 log accept in chain input in table filter in family ipv4 exists'}
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.check(chain='input', rule=_ru) == ret

def test_check_chain():
    if False:
        while True:
            i = 10
    '\n    Test if it check for the existence of a chain in the table\n    '
    assert nftables.check_chain() == {'result': False, 'comment': 'Chain needs to be specified'}
    mock = MagicMock(return_value='')
    ret = {'comment': 'Chain input in table filter in family ipv4 does not exist', 'result': False}
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.check_chain(chain='input') == ret
    mock = MagicMock(return_value='chain input {{')
    ret = {'comment': 'Chain input in table filter in family ipv4 exists', 'result': True}
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.check_chain(chain='input') == ret

def test_check_table():
    if False:
        print('Hello World!')
    '\n    Test if it check for the existence of a table\n    '
    assert nftables.check_table() == {'result': False, 'comment': 'Table needs to be specified'}
    mock = MagicMock(return_value='')
    ret = {'comment': 'Table nat in family ipv4 does not exist', 'result': False}
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.check_table(table='nat') == ret
    mock = MagicMock(return_value='table ip nat')
    ret = {'comment': 'Table nat in family ipv4 exists', 'result': True}
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.check_table(table='nat') == ret

def test_new_table():
    if False:
        print('Hello World!')
    '\n    Test if it create new custom table.\n    '
    assert nftables.new_table(table=None) == {'result': False, 'comment': 'Table needs to be specified'}
    mock = MagicMock(return_value='')
    ret = {'comment': 'Table nat in family ipv4 created', 'result': True}
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.new_table(table='nat') == ret
    mock = MagicMock(return_value='table ip nat')
    ret = {'comment': 'Table nat in family ipv4 exists', 'result': True}
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.new_table(table='nat') == ret

def test_delete_table():
    if False:
        return 10
    '\n    Test if it delete custom table.\n    '
    assert nftables.delete_table(table=None) == {'result': False, 'comment': 'Table needs to be specified'}
    mock_ret = {'result': False, 'comment': 'Table nat in family ipv4 does not exist'}
    with patch('salt.modules.nftables.check_table', MagicMock(return_value=mock_ret)):
        ret = nftables.delete_table(table='nat')
        assert ret == {'result': False, 'comment': 'Table nat in family ipv4 does not exist'}
    mock = MagicMock(return_value='table ip nat')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}), patch('salt.modules.nftables.check_table', MagicMock(return_value={'result': True, 'comment': ''})):
        assert nftables.delete_table(table='nat') == {'comment': 'Table nat in family ipv4 could not be deleted', 'result': False}
    mock = MagicMock(return_value='')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}), patch('salt.modules.nftables.check_table', MagicMock(return_value={'result': True, 'comment': ''})):
        assert nftables.delete_table(table='nat') == {'comment': 'Table nat in family ipv4 deleted', 'result': True}

def test_new_chain():
    if False:
        i = 10
        return i + 15
    '\n    Test if it create new chain to the specified table.\n    '
    assert nftables.new_chain() == {'result': False, 'comment': 'Chain needs to be specified'}
    ret = {'result': False, 'comment': 'Table filter in family ipv4 does not exist'}
    mock = MagicMock(return_value='')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.new_chain(chain='input') == ret
    ret = {'result': False, 'comment': 'Chain input in table filter in family ipv4 already exists'}
    mock = MagicMock(return_value='table ip filter chain input {{')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.new_chain(chain='input') == ret

def test_new_chain_variable():
    if False:
        return 10
    '\n    Test if it create new chain to the specified table.\n    '
    mock = MagicMock(return_value='')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}), patch('salt.modules.nftables.check_chain', MagicMock(return_value={'result': False, 'comment': ''})), patch('salt.modules.nftables.check_table', MagicMock(return_value={'result': True, 'comment': ''})):
        assert nftables.new_chain(chain='input', table_type='filter') == {'result': False, 'comment': 'Table_type, hook, and priority required.'}
        assert nftables.new_chain(chain='input', table_type='filter', hook='input', priority=0)

def test_delete_chain():
    if False:
        while True:
            i = 10
    '\n    Test if it delete the chain from the specified table.\n    '
    assert nftables.delete_chain() == {'result': False, 'comment': 'Chain needs to be specified'}
    ret = {'result': False, 'comment': 'Table filter in family ipv4 does not exist'}
    mock = MagicMock(return_value='')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.delete_chain(chain='input') == ret
    ret = {'result': False, 'comment': 'Chain input in table filter in family ipv4 could not be deleted'}
    mock = MagicMock(return_value='table ip filter')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}), patch('salt.modules.nftables.check_table', MagicMock(return_value={'result': True, 'comment': ''})), patch('salt.modules.nftables.check_chain', MagicMock(return_value={'result': True, 'comment': ''})):
        assert nftables.delete_chain(chain='input') == ret
    ret = {'result': True, 'comment': 'Chain input in table filter in family ipv4 deleted'}
    mock = MagicMock(return_value='')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}), patch('salt.modules.nftables.check_table', MagicMock(return_value={'result': True, 'comment': ''})), patch('salt.modules.nftables.check_chain', MagicMock(return_value={'result': True, 'comment': ''})):
        assert nftables.delete_chain(chain='input') == ret

def test_delete_chain_variables():
    if False:
        return 10
    '\n    Test if it delete the chain from the specified table.\n    '
    mock = MagicMock(return_value='')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}), patch('salt.modules.nftables.check_chain', MagicMock(return_value={'result': True, 'comment': ''})), patch('salt.modules.nftables.check_table', MagicMock(return_value={'result': True, 'comment': ''})):
        _expected = {'comment': 'Chain input in table filter in family ipv4 deleted', 'result': True}
        assert nftables.delete_chain(chain='input') == _expected

def test_append():
    if False:
        while True:
            i = 10
    '\n    Test if it append a rule to the specified table & chain.\n    '
    assert nftables.append() == {'result': False, 'comment': 'Chain needs to be specified'}
    assert nftables.append(chain='input') == {'result': False, 'comment': 'Rule needs to be specified'}
    _ru = 'input tcp dport 22 log accept'
    ret = {'comment': 'Table filter in family ipv4 does not exist', 'result': False}
    mock = MagicMock(return_value='')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.append(chain='input', rule=_ru) == ret
    ret = {'comment': 'Chain input in table filter in family ipv4 does not exist', 'result': False}
    mock = MagicMock(return_value='table ip filter')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.append(chain='input', rule=_ru) == ret
    r_val = 'table ip filter chain input {{ input tcp dport 22 log accept #'
    mock = MagicMock(return_value=r_val)
    _expected = {'comment': 'Rule input tcp dport 22 log accept chain input in table filter in family ipv4 already exists', 'result': False}
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.append(chain='input', rule=_ru) == _expected

def test_append_rule():
    if False:
        i = 10
        return i + 15
    '\n    Test if it append a rule to the specified table & chain.\n    '
    _ru = 'input tcp dport 22 log accept'
    mock = MagicMock(side_effect=['1', ''])
    with patch.dict(nftables.__salt__, {'cmd.run': mock}), patch('salt.modules.nftables.check', MagicMock(return_value={'result': False, 'comment': ''})), patch('salt.modules.nftables.check_chain', MagicMock(return_value={'result': True, 'comment': ''})), patch('salt.modules.nftables.check_table', MagicMock(return_value={'result': True, 'comment': ''})):
        _expected = {'comment': 'Failed to add rule "{}" chain input in table filter in family ipv4.'.format(_ru), 'result': False}
        assert nftables.append(chain='input', rule=_ru) == _expected
        _expected = {'comment': 'Added rule "{}" chain input in table filter in family ipv4.'.format(_ru), 'result': True}
        assert nftables.append(chain='input', rule=_ru) == _expected

def test_insert():
    if False:
        print('Hello World!')
    '\n    Test if it insert a rule into the specified table & chain,\n    at the specified position.\n    '
    assert nftables.insert() == {'result': False, 'comment': 'Chain needs to be specified'}
    assert nftables.insert(chain='input') == {'result': False, 'comment': 'Rule needs to be specified'}
    _ru = 'input tcp dport 22 log accept'
    ret = {'result': False, 'comment': 'Table filter in family ipv4 does not exist'}
    mock = MagicMock(return_value='')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.insert(chain='input', rule=_ru) == ret
    ret = {'result': False, 'comment': 'Chain input in table filter in family ipv4 does not exist'}
    mock = MagicMock(return_value='table ip filter')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.insert(chain='input', rule=_ru) == ret
    r_val = 'table ip filter chain input {{ input tcp dport 22 log accept #'
    mock = MagicMock(return_value=r_val)
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        res = nftables.insert(chain='input', rule=_ru)
        import logging
        log = logging.getLogger(__name__)
        log.debug('=== res %s ===', res)
        assert nftables.insert(chain='input', rule=_ru)

def test_insert_rule():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it insert a rule into the specified table & chain,\n    at the specified position.\n    '
    _ru = 'input tcp dport 22 log accept'
    mock = MagicMock(side_effect=['1', ''])
    with patch.dict(nftables.__salt__, {'cmd.run': mock}), patch('salt.modules.nftables.check', MagicMock(return_value={'result': False, 'comment': ''})), patch('salt.modules.nftables.check_chain', MagicMock(return_value={'result': True, 'comment': ''})), patch('salt.modules.nftables.check_table', MagicMock(return_value={'result': True, 'comment': ''})):
        _expected = {'result': False, 'comment': 'Failed to add rule "{}" chain input in table filter in family ipv4.'.format(_ru)}
        assert nftables.insert(chain='input', rule=_ru) == _expected
        _expected = {'result': True, 'comment': 'Added rule "{}" chain input in table filter in family ipv4.'.format(_ru)}
        assert nftables.insert(chain='input', rule=_ru) == _expected

def test_delete():
    if False:
        while True:
            i = 10
    "\n    Test if it delete a rule from the specified table & chain,\n    specifying either the rule in its entirety, or\n    the rule's position in the chain.\n    "
    _ru = 'input tcp dport 22 log accept'
    ret = {'result': False, 'comment': 'Only specify a position or a rule, not both'}
    assert nftables.delete(table='filter', chain='input', position='3', rule=_ru) == ret
    ret = {'result': False, 'comment': 'Table filter in family ipv4 does not exist'}
    mock = MagicMock(return_value='')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.delete(table='filter', chain='input', rule=_ru) == ret
    ret = {'result': False, 'comment': 'Chain input in table filter in family ipv4 does not exist'}
    mock = MagicMock(return_value='table ip filter')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.delete(table='filter', chain='input', rule=_ru) == ret
    mock = MagicMock(return_value='table ip filter chain input {{')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.delete(table='filter', chain='input', rule=_ru)

def test_delete_rule():
    if False:
        print('Hello World!')
    "\n    Test if it delete a rule from the specified table & chain,\n    specifying either the rule in its entirety, or\n    the rule's position in the chain.\n    "
    mock = MagicMock(side_effect=['1', ''])
    with patch.dict(nftables.__salt__, {'cmd.run': mock}), patch('salt.modules.nftables.check', MagicMock(return_value={'result': True, 'comment': ''})), patch('salt.modules.nftables.check_chain', MagicMock(return_value={'result': True, 'comment': ''})), patch('salt.modules.nftables.check_table', MagicMock(return_value={'result': True, 'comment': ''})):
        _expected = {'result': False, 'comment': 'Failed to delete rule "None" in chain input  table filter in family ipv4'}
        assert nftables.delete(table='filter', chain='input', position='3') == _expected
        _expected = {'result': True, 'comment': 'Deleted rule "None" in chain input in table filter in family ipv4.'}
        assert nftables.delete(table='filter', chain='input', position='3') == _expected

def test_flush():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test if it flush the chain in the specified table, flush all chains\n    in the specified table if chain is not specified.\n    '
    ret = {'result': False, 'comment': 'Table filter in family ipv4 does not exist'}
    mock = MagicMock(return_value='')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.flush(table='filter', chain='input') == ret
    ret = {'result': False, 'comment': 'Chain input in table filter in family ipv4 does not exist'}
    mock = MagicMock(return_value='table ip filter')
    with patch.dict(nftables.__salt__, {'cmd.run': mock}):
        assert nftables.flush(table='filter', chain='input') == ret

def test_flush_chain():
    if False:
        return 10
    '\n    Test if it flush the chain in the specified table, flush all chains\n    in the specified table if chain is not specified.\n    '
    mock = MagicMock(side_effect=['1', ''])
    with patch.dict(nftables.__salt__, {'cmd.run': mock}), patch('salt.modules.nftables.check_chain', MagicMock(return_value={'result': True, 'comment': ''})), patch('salt.modules.nftables.check_table', MagicMock(return_value={'result': True, 'comment': ''})):
        _expected = {'result': False, 'comment': 'Failed to flush rules from chain input in table filter in family ipv4.'}
        assert nftables.flush(table='filter', chain='input') == _expected
        _expected = {'result': True, 'comment': 'Flushed rules from chain input in table filter in family ipv4.'}
        assert nftables.flush(table='filter', chain='input') == _expected

def test_get_policy():
    if False:
        return 10
    '\n    Test the current policy for the specified table/chain\n    '
    list_rules_return = '\n    {\n      "nftables": [\n        {\n          "table": {\n            "family": "ip",\n            "name": "filter",\n            "handle": 47\n          }\n        },\n        {\n          "chain": {\n            "family": "ip",\n            "table": "filter",\n            "name": "input",\n            "handle": 1,\n            "type": "filter",\n            "hook": "input",\n            "prio": 0,\n            "policy": "accept"\n          }\n        },\n        {\n          "chain": {\n            "family": "ip",\n            "table": "filter",\n            "name": "forward",\n            "handle": 2,\n            "type": "filter",\n            "hook": "forward",\n            "prio": 0,\n            "policy": "accept"\n          }\n        },\n        {\n          "chain": {\n            "family": "ip",\n            "table": "filter",\n            "name": "output",\n            "handle": 3,\n            "type": "filter",\n            "hook": "output",\n            "prio": 0,\n            "policy": "accept"\n          }\n        }\n      ]\n    }\n    '
    expected = json.loads(list_rules_return)
    assert nftables.get_policy(table='filter', chain=None, family='ipv4') == 'Error: Chain needs to be specified'
    with patch.object(nftables, 'get_rules_json', MagicMock(return_value=expected)):
        assert nftables.get_policy(table='filter', chain='input', family='ipv4') == 'accept'
    with patch.object(nftables, 'get_rules_json', MagicMock(return_value=expected)):
        assert nftables.get_policy(table='filter', chain='missing', family='ipv4') is None

def test_set_policy():
    if False:
        print('Hello World!')
    '\n    Test set the current policy for the specified table/chain\n    '
    list_rules_return = '\n    {\n      "nftables": [\n        {\n          "table": {\n            "family": "ip",\n            "name": "filter",\n            "handle": 47\n          }\n        },\n        {\n          "chain": {\n            "family": "ip",\n            "table": "filter",\n            "name": "input",\n            "handle": 1,\n            "type": "filter",\n            "hook": "input",\n            "prio": 0,\n            "policy": "accept"\n          }\n        },\n        {\n          "chain": {\n            "family": "ip",\n            "table": "filter",\n            "name": "forward",\n            "handle": 2,\n            "type": "filter",\n            "hook": "forward",\n            "prio": 0,\n            "policy": "accept"\n          }\n        },\n        {\n          "chain": {\n            "family": "ip",\n            "table": "filter",\n            "name": "output",\n            "handle": 3,\n            "type": "filter",\n            "hook": "output",\n            "prio": 0,\n            "policy": "accept"\n          }\n        }\n      ]\n    }\n    '
    expected = json.loads(list_rules_return)['nftables']
    assert nftables.set_policy(table='filter', chain=None, policy=None, family='ipv4') == 'Error: Chain needs to be specified'
    assert nftables.set_policy(table='filter', chain='input', policy=None, family='ipv4') == 'Error: Policy needs to be specified'
    mock = MagicMock(return_value={'retcode': 0})
    with patch.object(nftables, 'get_rules_json', MagicMock(return_value=expected)):
        with patch.dict(nftables.__salt__, {'cmd.run_all': mock}):
            assert nftables.set_policy(table='filter', chain='input', policy='accept', family='ipv4')