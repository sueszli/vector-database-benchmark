import pytest
import salt.utils.win_dotnet as win_dotnet
from tests.support.mock import MagicMock, patch
from tests.support.unit import TestCase
pytestmark = [pytest.mark.skip_unless_on_windows]

class WinDotNetTestCase(TestCase):
    """
    Test cases for salt.utils.win_dotnet
    """

    def setUp(self):
        if False:
            return 10
        self.mock_reg_list = MagicMock(return_value=['CDF', 'v2.0.50727', 'v3.0', 'v3.5', 'v4', 'v4.0'])
        self.mock_reg_exists = MagicMock(side_effect=[True, True, True, False, True, False, False])
        self.mock_reg_read = MagicMock(side_effect=[{'vdata': 1}, {'vdata': '2.0.50727.4927'}, {'vdata': 2}, {'vdata': 1}, {'vdata': '3.0.30729.4926'}, {'vdata': 2}, {'vdata': 1}, {'vdata': '3.5.30729.4926'}, {'vdata': 1}, {'vdata': 1}, {'vdata': 461814}])

    def test_versions(self):
        if False:
            return 10
        '\n        Test the versions function\n        '
        expected = {'details': {'v2.0.50727': {'full': '2.0.50727.4927 SP2', 'service_pack': 2, 'version': '2.0.50727.4927'}, 'v3.0': {'full': '3.0.30729.4926 SP2', 'service_pack': 2, 'version': '3.0.30729.4926'}, 'v3.5': {'full': '3.5.30729.4926 SP1', 'service_pack': 1, 'version': '3.5.30729.4926'}, 'v4': {'full': '4.7.2', 'service_pack': 'N/A', 'version': '4.7.2'}}, 'versions': ['2.0.50727.4927', '3.0.30729.4926', '3.5.30729.4926', '4.7.2']}
        with patch('salt.utils.win_reg.list_keys', self.mock_reg_list), patch('salt.utils.win_reg.value_exists', self.mock_reg_exists), patch('salt.utils.win_reg.read_value', self.mock_reg_read):
            result = win_dotnet.versions()
            self.assertDictEqual(result, expected)

    def test_versions_list(self):
        if False:
            while True:
                i = 10
        expected = ['2.0.50727.4927', '3.0.30729.4926', '3.5.30729.4926', '4.7.2']
        with patch('salt.utils.win_reg.list_keys', self.mock_reg_list), patch('salt.utils.win_reg.value_exists', self.mock_reg_exists), patch('salt.utils.win_reg.read_value', self.mock_reg_read):
            result = win_dotnet.versions_list()
            self.assertListEqual(result, expected)

    def test_versions_details(self):
        if False:
            i = 10
            return i + 15
        '\n        Test the versions function\n        '
        expected = {'v2.0.50727': {'full': '2.0.50727.4927 SP2', 'service_pack': 2, 'version': '2.0.50727.4927'}, 'v3.0': {'full': '3.0.30729.4926 SP2', 'service_pack': 2, 'version': '3.0.30729.4926'}, 'v3.5': {'full': '3.5.30729.4926 SP1', 'service_pack': 1, 'version': '3.5.30729.4926'}, 'v4': {'full': '4.7.2', 'service_pack': 'N/A', 'version': '4.7.2'}}
        with patch('salt.utils.win_reg.list_keys', self.mock_reg_list), patch('salt.utils.win_reg.value_exists', self.mock_reg_exists), patch('salt.utils.win_reg.read_value', self.mock_reg_read):
            result = win_dotnet.versions_details()
            self.assertDictEqual(result, expected)

    def test_version_atleast_35(self):
        if False:
            return 10
        with patch('salt.utils.win_reg.list_keys', self.mock_reg_list), patch('salt.utils.win_reg.value_exists', self.mock_reg_exists), patch('salt.utils.win_reg.read_value', self.mock_reg_read):
            self.assertTrue(win_dotnet.version_at_least('3.5'))

    def test_version_atleast_47(self):
        if False:
            print('Hello World!')
        with patch('salt.utils.win_reg.list_keys', self.mock_reg_list), patch('salt.utils.win_reg.value_exists', self.mock_reg_exists), patch('salt.utils.win_reg.read_value', self.mock_reg_read):
            self.assertTrue(win_dotnet.version_at_least('4.7'))

    def test_version_atleast_472(self):
        if False:
            return 10
        with patch('salt.utils.win_reg.list_keys', self.mock_reg_list), patch('salt.utils.win_reg.value_exists', self.mock_reg_exists), patch('salt.utils.win_reg.read_value', self.mock_reg_read):
            self.assertTrue(win_dotnet.version_at_least('4.7.2'))

    def test_version_not_atleast_473(self):
        if False:
            while True:
                i = 10
        with patch('salt.utils.win_reg.list_keys', self.mock_reg_list), patch('salt.utils.win_reg.value_exists', self.mock_reg_exists), patch('salt.utils.win_reg.read_value', self.mock_reg_read):
            self.assertFalse(win_dotnet.version_at_least('4.7.3'))