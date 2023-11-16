from pathlib import PurePath
import pytest
from vyper.cli.vyper_json import TRANSLATE_MAP, get_output_formats
from vyper.exceptions import JSONError

def test_no_outputs():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(KeyError):
        get_output_formats({}, {})

def test_invalid_output():
    if False:
        while True:
            i = 10
    input_json = {'settings': {'outputSelection': {'foo.vy': ['abi', 'foobar']}}}
    targets = [PurePath('foo.vy')]
    with pytest.raises(JSONError):
        get_output_formats(input_json, targets)

def test_unknown_contract():
    if False:
        i = 10
        return i + 15
    input_json = {'settings': {'outputSelection': {'bar.vy': ['abi']}}}
    targets = [PurePath('foo.vy')]
    with pytest.raises(JSONError):
        get_output_formats(input_json, targets)

@pytest.mark.parametrize('output', TRANSLATE_MAP.items())
def test_translate_map(output):
    if False:
        while True:
            i = 10
    input_json = {'settings': {'outputSelection': {'foo.vy': [output[0]]}}}
    targets = [PurePath('foo.vy')]
    assert get_output_formats(input_json, targets) == {PurePath('foo.vy'): [output[1]]}

def test_star():
    if False:
        print('Hello World!')
    input_json = {'settings': {'outputSelection': {'*': ['*']}}}
    targets = [PurePath('foo.vy'), PurePath('bar.vy')]
    expected = sorted(set(TRANSLATE_MAP.values()))
    result = get_output_formats(input_json, targets)
    assert result == {PurePath('foo.vy'): expected, PurePath('bar.vy'): expected}

def test_evm():
    if False:
        for i in range(10):
            print('nop')
    input_json = {'settings': {'outputSelection': {'foo.vy': ['abi', 'evm']}}}
    targets = [PurePath('foo.vy')]
    expected = ['abi'] + sorted((v for (k, v) in TRANSLATE_MAP.items() if k.startswith('evm')))
    result = get_output_formats(input_json, targets)
    assert result == {PurePath('foo.vy'): expected}

def test_solc_style():
    if False:
        return 10
    input_json = {'settings': {'outputSelection': {'foo.vy': {'': ['abi'], 'foo.vy': ['ir']}}}}
    targets = [PurePath('foo.vy')]
    assert get_output_formats(input_json, targets) == {PurePath('foo.vy'): ['abi', 'ir_dict']}

def test_metadata():
    if False:
        while True:
            i = 10
    input_json = {'settings': {'outputSelection': {'*': ['metadata']}}}
    targets = [PurePath('foo.vy')]
    assert get_output_formats(input_json, targets) == {PurePath('foo.vy'): ['metadata']}