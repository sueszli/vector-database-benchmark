import pytest
from salt.modules import boto_ssm
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        for i in range(10):
            print('nop')
    return {boto_ssm: {'__utils__': {'boto3.assign_funcs': MagicMock()}}}

def test___virtual_has_boto_reqs_true():
    if False:
        i = 10
        return i + 15
    with patch('salt.utils.versions.check_boto_reqs', return_value=True):
        result = boto_ssm.__virtual__()
    assert result is True

def test___virtual_has_boto_reqs_false():
    if False:
        for i in range(10):
            print('nop')
    with patch('salt.utils.versions.check_boto_reqs', return_value=False):
        result = boto_ssm.__virtual__()
    assert result is False