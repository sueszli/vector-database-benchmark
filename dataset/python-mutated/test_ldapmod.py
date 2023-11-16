"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.ldapmod
"""
import time
import pytest
import salt.modules.ldapmod as ldapmod
from tests.support.mock import MagicMock, patch

@pytest.fixture
def configure_loader_modules():
    if False:
        for i in range(10):
            print('nop')
    return {ldapmod: {}}

def test_search():
    if False:
        while True:
            i = 10
    '\n    Test if it run an arbitrary LDAP query and return the results.\n    '

    class MockConnect:
        """
        Mocking _connect method
        """

        def __init__(self):
            if False:
                return 10
            self.bdn = None
            self.scope = None
            self._filter = None
            self.attrs = None

        def search_s(self, bdn, scope, _filter, attrs):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Mock function for search_s\n            '
            self.bdn = bdn
            self.scope = scope
            self._filter = _filter
            self.attrs = attrs
            return 'SALT'
    mock = MagicMock(return_value=True)
    with patch.dict(ldapmod.__salt__, {'config.option': mock}):
        with patch.object(ldapmod, '_connect', MagicMock(return_value=MockConnect())):
            with patch.object(time, 'time', MagicMock(return_value=0.0008)):
                assert ldapmod.search(filter='myhost') == {'count': 4, 'results': 'SALT', 'time': {'raw': '0.0', 'human': '0.0ms'}}