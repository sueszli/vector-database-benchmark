"""
Unit tests for the LGPO module
"""
import platform
import pytest
from packaging import version
pytestmark = [pytest.mark.windows_whitelisted, pytest.mark.skip_unless_on_windows, pytest.mark.slow_test]

def _parse_platform_version():
    if False:
        print('Hello World!')
    if platform.system() != 'Windows':
        return True
    return version.parse(platform.version()) < version.parse('10.0.16299')

@pytest.fixture(scope='module')
def lgpo(modules):
    if False:
        for i in range(10):
            print('nop')
    return modules.lgpo

@pytest.mark.skipif(_parse_platform_version(), reason='Policy only available on 10.0.16299 or later')
def test_62058_whitespace(lgpo):
    if False:
        for i in range(10):
            print('nop')
    result = lgpo.get_policy_info('Allow Online Tips', 'machine')
    for element in result['policy_elements']:
        if 'element_aliases' in element:
            assert 'Allow Settings to retrieve online tips.' in element['element_aliases']
            return
    assert False

def test_61859(lgpo):
    if False:
        for i in range(10):
            print('nop')
    expected = 'ADMX policy name/id "Pol_CipherSuiteOrder" is used in multiple ADMX files.\nTry one of the following names:\n - Lanman Server\\Network\\Cipher suite order\n - Lanman Workstation\\Network\\Cipher suite order'
    result = lgpo.get_policy_info(policy_name='Pol_CipherSuiteOrder', policy_class='machine')
    assert result['message'] == expected