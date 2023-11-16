from __future__ import annotations
import re
from unittest import mock
import pytest
from airflow.exceptions import AirflowConfigException
from airflow.utils import net
from tests.test_utils.config import conf_vars

def get_hostname():
    if False:
        return 10
    return 'awesomehostname'

class TestGetHostname:

    @mock.patch('airflow.utils.net.getfqdn', return_value='first')
    @conf_vars({('core', 'hostname_callable'): None})
    def test_get_hostname_unset(self, mock_getfqdn):
        if False:
            for i in range(10):
                print('nop')
        assert 'first' == net.get_hostname()

    @conf_vars({('core', 'hostname_callable'): 'tests.utils.test_net.get_hostname'})
    def test_get_hostname_set(self):
        if False:
            for i in range(10):
                print('nop')
        assert 'awesomehostname' == net.get_hostname()

    @conf_vars({('core', 'hostname_callable'): 'tests.utils.test_net'})
    def test_get_hostname_set_incorrect(self):
        if False:
            return 10
        with pytest.raises(TypeError):
            net.get_hostname()

    @conf_vars({('core', 'hostname_callable'): 'tests.utils.test_net.missing_func'})
    def test_get_hostname_set_missing(self):
        if False:
            print('Hello World!')
        with pytest.raises(AirflowConfigException, match=re.escape('The object could not be loaded. Please check "hostname_callable" key in "core" section. Current value: "tests.utils.test_net.missing_func"')):
            net.get_hostname()