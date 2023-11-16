"""
unittests for json outputter
"""
import pytest
import salt.output.json_out as json_out
import salt.utils.stringutils
from tests.support.mock import patch

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {json_out: {}}

@pytest.fixture
def data():
    if False:
        return 10
    return {'test': 'two', 'example': 'one'}

def test_default_output(data):
    if False:
        i = 10
        return i + 15
    ret = json_out.output(data)
    assert '"test": "two"' in ret
    assert '"example": "one"' in ret

def test_pretty_output(data):
    if False:
        while True:
            i = 10
    with patch.dict(json_out.__opts__, {'output_indent': 'pretty'}):
        ret = json_out.output(data)
        assert '"test": "two"' in ret
        assert '"example": "one"' in ret

def test_indent_output(data):
    if False:
        print('Hello World!')
    with patch.dict(json_out.__opts__, {'output_indent': 2}):
        ret = json_out.output(data)
        assert '"test": "two"' in ret
        assert '"example": "one"' in ret

def test_negative_zero_output(data):
    if False:
        i = 10
        return i + 15
    with patch.dict(json_out.__opts__, {'output_indent': 0}):
        ret = json_out.output(data)
        assert '"test": "two"' in ret
        assert '"example": "one"' in ret

def test_negative_int_output(data):
    if False:
        i = 10
        return i + 15
    with patch.dict(json_out.__opts__, {'output_indent': -1}):
        ret = json_out.output(data)
        assert '"test": "two"' in ret
        assert '"example": "one"' in ret

def test_unicode_output():
    if False:
        for i in range(10):
            print('nop')
    with patch.dict(json_out.__opts__, {'output_indent': 'pretty'}):
        decoded = {'test': 'Д', 'example': 'one'}
        encoded = {'test': salt.utils.stringutils.to_str('Д'), 'example': 'one'}
        expected = '{\n    "example": "one",\n    "test": "Д"\n}'
        assert json_out.output(decoded) == expected
        assert json_out.output(encoded) == expected