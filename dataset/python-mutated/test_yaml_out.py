"""
unittests for yaml outputter
"""
import pytest
import salt.output.yaml_out as yaml
from tests.support.mock import patch

@pytest.fixture
def configure_loader_modules():
    if False:
        i = 10
        return i + 15
    return {yaml: {}}

@pytest.fixture
def data():
    if False:
        while True:
            i = 10
    return {'test': 'two', 'example': 'one'}

def test_default_output(data):
    if False:
        for i in range(10):
            print('nop')
    ret = yaml.output(data)
    expect = 'example: one\ntest: two\n'
    assert expect == ret

def test_negative_int_output(data):
    if False:
        return 10
    with patch.dict(yaml.__opts__, {'output_indent': -1}):
        ret = yaml.output(data)
        expect = '{example: one, test: two}\n'
        assert expect == ret