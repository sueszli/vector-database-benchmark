import re
import pytest
from allennlp.version import VERSION
VALID_VERSION_RE = re.compile('^(0|[1-9]\\d*)\\.(0|[1-9]\\d*)\\.(0|[1-9]\\d*)(rc(0|[1-9]\\d*))?(\\.post(0|[1-9]\\d*))?(\\.dev2020[0-9]{4})?$')

def is_valid(version: str) -> bool:
    if False:
        while True:
            i = 10
    return VALID_VERSION_RE.match(version) is not None

@pytest.mark.parametrize('version, valid', [('1.0.0', True), ('1.0.0rc3', True), ('1.0.0.post0', True), ('1.0.0.post1', True), ('1.0.0rc3.post0', True), ('1.0.0rc3.post0.dev20200424', True), ('1.0.0.rc3', False), ('1.0.0rc01', False), ('1.0.0rc3.dev2020424', False)])
def test_is_valid_helper(version: str, valid: bool):
    if False:
        print('Hello World!')
    assert is_valid(version) is valid

def test_version():
    if False:
        return 10
    '\n    Ensures current version is consistent with our conventions.\n    '
    assert is_valid(VERSION)