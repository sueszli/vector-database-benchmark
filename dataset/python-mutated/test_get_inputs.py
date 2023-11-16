from pathlib import PurePath
import pytest
from vyper.cli.vyper_json import get_compilation_targets, get_inputs
from vyper.exceptions import JSONError
from vyper.utils import keccak256
FOO_CODE = '\nimport contracts.bar as Bar\n\n@external\ndef foo(a: address) -> bool:\n    return Bar(a).bar(1)\n'
BAR_CODE = '\n@external\ndef bar(a: uint256) -> bool:\n    return True\n'

def test_no_sources():
    if False:
        while True:
            i = 10
    with pytest.raises(KeyError):
        get_inputs({})

def test_contracts_urls():
    if False:
        while True:
            i = 10
    with pytest.raises(JSONError):
        get_inputs({'sources': {'foo.vy': {'urls': ['https://foo.code.com/']}}})

def test_contracts_no_content_key():
    if False:
        i = 10
        return i + 15
    with pytest.raises(JSONError):
        get_inputs({'sources': {'foo.vy': FOO_CODE}})

def test_contracts_keccak():
    if False:
        i = 10
        return i + 15
    hash_ = keccak256(FOO_CODE.encode()).hex()
    input_json = {'sources': {'foo.vy': {'content': FOO_CODE, 'keccak256': hash_}}}
    get_inputs(input_json)
    input_json['sources']['foo.vy']['keccak256'] = '0x' + hash_
    get_inputs(input_json)
    input_json['sources']['foo.vy']['keccak256'] = '0x1234567890'
    with pytest.raises(JSONError):
        get_inputs(input_json)

def test_contracts_outside_pwd():
    if False:
        i = 10
        return i + 15
    input_json = {'sources': {'../foo.vy': {'content': FOO_CODE}}}
    get_inputs(input_json)

def test_contract_collision():
    if False:
        i = 10
        return i + 15
    input_json = {'sources': {'./foo.vy': {'content': FOO_CODE}, 'foo.vy': {'content': FOO_CODE}}}
    with pytest.raises(JSONError):
        get_inputs(input_json)

def test_contracts_return_value():
    if False:
        return 10
    input_json = {'sources': {'foo.vy': {'content': FOO_CODE}, 'contracts/bar.vy': {'content': BAR_CODE}}}
    result = get_inputs(input_json)
    assert result == {PurePath('foo.vy'): {'content': FOO_CODE}, PurePath('contracts/bar.vy'): {'content': BAR_CODE}}
BAR_ABI = [{'name': 'bar', 'outputs': [{'type': 'bool', 'name': 'out'}], 'inputs': [{'type': 'uint256', 'name': 'a'}], 'stateMutability': 'nonpayable', 'type': 'function'}]

def test_interface_collision():
    if False:
        while True:
            i = 10
    input_json = {'sources': {'foo.vy': {'content': FOO_CODE}}, 'interfaces': {'bar.json': {'abi': BAR_ABI}, 'bar.vy': {'content': BAR_CODE}}}
    with pytest.raises(JSONError):
        get_inputs(input_json)

def test_json_no_abi():
    if False:
        return 10
    input_json = {'sources': {'foo.vy': {'content': FOO_CODE}}, 'interfaces': {'bar.json': {'content': BAR_ABI}}}
    with pytest.raises(JSONError):
        get_inputs(input_json)

def test_vy_no_content():
    if False:
        for i in range(10):
            print('nop')
    input_json = {'sources': {'foo.vy': {'content': FOO_CODE}}, 'interfaces': {'bar.vy': {'abi': BAR_CODE}}}
    with pytest.raises(JSONError):
        get_inputs(input_json)

def test_interfaces_output():
    if False:
        return 10
    input_json = {'sources': {'foo.vy': {'content': FOO_CODE}}, 'interfaces': {'bar.json': {'abi': BAR_ABI}, 'interface.folder/bar2.vy': {'content': BAR_CODE}}}
    targets = get_compilation_targets(input_json)
    assert targets == [PurePath('foo.vy')]
    result = get_inputs(input_json)
    assert result == {PurePath('foo.vy'): {'content': FOO_CODE}, PurePath('bar.json'): {'abi': BAR_ABI}, PurePath('interface.folder/bar2.vy'): {'content': BAR_CODE}}

@pytest.mark.xfail
def test_manifest_output():
    if False:
        while True:
            i = 10
    input_json = {'interfaces': {'bar.json': {'contractTypes': {'Bar': {'abi': BAR_ABI}}}}}
    result = get_inputs(input_json)
    assert isinstance(result, dict)
    assert result == {'Bar': {'type': 'json', 'code': BAR_ABI}}