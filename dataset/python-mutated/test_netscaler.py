"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>
"""
import salt.modules.netscaler as netscaler
from tests.support.mixins import LoaderModuleMockMixin
from tests.support.mock import MagicMock, patch
from tests.support.unit import TestCase

class MockJson(Exception):
    """
    Mock Json class
    """

    @staticmethod
    def loads(content):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock loads method\n        '
        return content

    @staticmethod
    def dumps(dumps):
        if False:
            return 10
        '\n        Mock dumps method\n        '
        return dumps

class MockNSNitroError(Exception):
    """
    Mock NSNitroError class
    """

    def __init__(self, message='error'):
        if False:
            for i in range(10):
                print('nop')
        self._message = message
        super().__init__(self.message)

    def _get_message(self):
        if False:
            i = 10
            return i + 15
        '\n        get_message method\n        '
        return self._message

    def _set_message(self, message):
        if False:
            return 10
        '\n        set_message method\n        '
        self._message = message
    message = property(_get_message, _set_message)

class MockNSNitro:
    """
    Mock NSNitro class
    """
    flag = None

    def __init__(self, host, user, passwd, bol):
        if False:
            print('Hello World!')
        pass

    @staticmethod
    def login():
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock login method\n        '
        return True

    @staticmethod
    def logout():
        if False:
            return 10
        '\n        Mock logout method\n        '
        return True

class MockNSServiceGroup:
    """
    Mock NSServiceGroup class
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.sg_name = None

    def set_servicegroupname(self, sg_name):
        if False:
            return 10
        '\n        Mock set_servicegroupname method\n        '
        self.sg_name = sg_name
        return MockNSServiceGroup()

    @staticmethod
    def get(obj, servicegroup):
        if False:
            print('Hello World!')
        '\n        Mock get method\n        '
        if MockNSNitro.flag:
            raise MockNSNitroError
        return MockNSServiceGroup()

    @staticmethod
    def add(obj, servicegroup):
        if False:
            i = 10
            return i + 15
        '\n        Mock add method\n        '
        if MockNSNitro.flag:
            raise MockNSNitroError
        return MockNSServiceGroup()

    @staticmethod
    def delete(obj, servicegroup):
        if False:
            print('Hello World!')
        '\n        Mock delete method\n        '
        if MockNSNitro.flag:
            raise MockNSNitroError
        return MockNSServiceGroup()

    @staticmethod
    def get_servers(obj, servicegroup):
        if False:
            while True:
                i = 10
        '\n        Mock get_servers method\n        '
        if MockNSNitro.flag:
            raise MockNSNitroError
        return [MockNSServiceGroup()]

    @staticmethod
    def enable_server(obj, servicegroup):
        if False:
            print('Hello World!')
        '\n        Mock enable_server method\n        '
        if MockNSNitro.flag:
            raise MockNSNitroError
        return MockNSServiceGroup()

    @staticmethod
    def disable_server(obj, servicegroup):
        if False:
            print('Hello World!')
        '\n        Mock disable_server method\n        '
        if MockNSNitro.flag:
            raise MockNSNitroError
        return MockNSServiceGroup()

    @staticmethod
    def get_servername():
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock get_servername method\n        '
        return 'serviceGroupName'

    @staticmethod
    def get_state():
        if False:
            print('Hello World!')
        '\n        Mock get_state method\n        '
        return 'ENABLED'

    @staticmethod
    def get_servicetype():
        if False:
            i = 10
            return i + 15
        '\n        Mock get_servicetype method\n        '
        return ''

    @staticmethod
    def set_servicetype(bol):
        if False:
            return 10
        '\n        Mock set_servicetype method\n        '
        return bol

class MockNSServiceGroupServerBinding:
    """
    Mock NSServiceGroupServerBinding class
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.sg_name = None

    def set_servername(self, sg_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock set_servername method\n        '
        self.sg_name = sg_name
        return MockNSServiceGroupServerBinding()

    def set_servicegroupname(self, sg_name):
        if False:
            while True:
                i = 10
        '\n        Mock set_servicegroupname method\n        '
        self.sg_name = sg_name
        return MockNSServiceGroupServerBinding()

    def set_port(self, sg_name):
        if False:
            while True:
                i = 10
        '\n        Mock set_port method\n        '
        self.sg_name = sg_name
        return MockNSServiceGroupServerBinding()

    @staticmethod
    def add(obj, servicegroup):
        if False:
            return 10
        '\n        Mock add method\n        '
        if MockNSNitro.flag:
            raise MockNSNitroError
        return MockNSServiceGroupServerBinding()

    @staticmethod
    def delete(obj, servicegroup):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock delete method\n        '
        if MockNSNitro.flag:
            raise MockNSNitroError
        return MockNSServiceGroupServerBinding()

class MockNSService:
    """
    Mock NSService class
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.sg_name = None

    def set_name(self, sg_name):
        if False:
            print('Hello World!')
        '\n        Mock set_name method\n        '
        self.sg_name = sg_name
        return MockNSService()

    @staticmethod
    def get(obj, servicegroup):
        if False:
            print('Hello World!')
        '\n        Mock get method\n        '
        if MockNSNitro.flag:
            raise MockNSNitroError
        return MockNSService()

    @staticmethod
    def enable(obj, servicegroup):
        if False:
            i = 10
            return i + 15
        '\n        Mock enable method\n        '
        if MockNSNitro.flag:
            raise MockNSNitroError
        return MockNSService()

    @staticmethod
    def disable(obj, servicegroup):
        if False:
            return 10
        '\n        Mock disable method\n        '
        if MockNSNitro.flag:
            raise MockNSNitroError
        return MockNSService()

    @staticmethod
    def get_svrstate():
        if False:
            return 10
        '\n        Mock get_svrstate method\n        '
        return 'UP'

class MockNSServer:
    """
    Mock NSServer class
    """
    flag = None

    def __init__(self):
        if False:
            while True:
                i = 10
        self.sg_name = None

    def set_name(self, sg_name):
        if False:
            while True:
                i = 10
        '\n        Mock set_name method\n        '
        self.sg_name = sg_name
        return MockNSServer()

    @staticmethod
    def get(obj, servicegroup):
        if False:
            while True:
                i = 10
        '\n        Mock get method\n        '
        return MockNSServer()

    @staticmethod
    def add(obj, servicegroup):
        if False:
            while True:
                i = 10
        '\n        Mock add method\n        '
        return MockNSServer()

    @staticmethod
    def delete(obj, servicegroup):
        if False:
            return 10
        '\n        Mock delete method\n        '
        return MockNSServer()

    @staticmethod
    def update(obj, servicegroup):
        if False:
            while True:
                i = 10
        '\n        Mock update method\n        '
        return MockNSServer()

    @staticmethod
    def enable(obj, servicegroup):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock enable method\n        '
        return MockNSServer()

    @staticmethod
    def disable(obj, servicegroup):
        if False:
            return 10
        '\n        Mock disable method\n        '
        return MockNSServer()

    @staticmethod
    def get_ipaddress():
        if False:
            while True:
                i = 10
        '\n        Mock get_ipaddress method\n        '
        return ''

    @staticmethod
    def set_ipaddress(s_ip):
        if False:
            return 10
        '\n        Mock set_ipaddress method\n        '
        return s_ip

    def get_state(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock get_state method\n        '
        if self.flag == 1:
            return ''
        elif self.flag == 2:
            return 'DISABLED'
        return 'ENABLED'

class MockNSLBVServer:
    """
    Mock NSLBVServer class
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.sg_name = None

    def set_name(self, sg_name):
        if False:
            return 10
        '\n        Mock set_name method\n        '
        self.sg_name = sg_name
        return MockNSLBVServer()

    @staticmethod
    def get(obj, servicegroup):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock get method\n        '
        return MockNSLBVServer()

    @staticmethod
    def set_ipv46(v_ip):
        if False:
            print('Hello World!')
        '\n        Mock set_ipv46 method\n        '
        return v_ip

    @staticmethod
    def set_port(v_port):
        if False:
            while True:
                i = 10
        '\n        Mock set_port method\n        '
        return v_port

    @staticmethod
    def set_servicetype(v_type):
        if False:
            return 10
        '\n        Mock set_servicetype method\n        '
        return v_type

    @staticmethod
    def get_ipv46():
        if False:
            while True:
                i = 10
        '\n        Mock get_ipv46 method\n        '
        return ''

    @staticmethod
    def get_port():
        if False:
            return 10
        '\n        Mock get_port method\n        '
        return ''

    @staticmethod
    def get_servicetype():
        if False:
            i = 10
            return i + 15
        '\n        Mock get_servicetype method\n        '
        return ''

    @staticmethod
    def add(obj, servicegroup):
        if False:
            while True:
                i = 10
        '\n        Mock add method\n        '
        return MockNSLBVServer()

    @staticmethod
    def delete(obj, servicegroup):
        if False:
            i = 10
            return i + 15
        '\n        Mock delete method\n        '
        return MockNSLBVServer()

class MockNSLBVServerServiceGroupBinding:
    """
    Mock NSLBVServerServiceGroupBinding class
    """
    flag = None

    def __init__(self):
        if False:
            while True:
                i = 10
        self.sg_name = None

    def set_name(self, sg_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock set_name method\n        '
        self.sg_name = sg_name
        return MockNSLBVServerServiceGroupBinding()

    @staticmethod
    def get(obj, servicegroup):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock get method\n        '
        if MockNSNitro.flag:
            raise MockNSNitroError
        return [MockNSLBVServerServiceGroupBinding()]

    @staticmethod
    def get_servicegroupname():
        if False:
            return 10
        '\n        Mock get_servicegroupname method\n        '
        return 'serviceGroupName'

    def set_servicegroupname(self, sg_name):
        if False:
            print('Hello World!')
        '\n        Mock set_servicegroupname method\n        '
        self.sg_name = sg_name
        if self.flag:
            return None
        return MockNSLBVServerServiceGroupBinding()

    @staticmethod
    def add(obj, servicegroup):
        if False:
            i = 10
            return i + 15
        '\n        Mock add method\n        '
        if MockNSNitro.flag:
            raise MockNSNitroError
        return MockNSLBVServerServiceGroupBinding()

    @staticmethod
    def delete(obj, servicegroup):
        if False:
            return 10
        '\n        Mock delete method\n        '
        if MockNSNitro.flag:
            raise MockNSNitroError
        return MockNSLBVServerServiceGroupBinding()

class MockNSSSLVServerSSLCertKeyBinding:
    """
    Mock NSSSLVServerSSLCertKeyBinding class
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.sg_name = None

    def set_vservername(self, sg_name):
        if False:
            while True:
                i = 10
        '\n        Mock set_vservername method\n        '
        self.sg_name = sg_name
        return MockNSSSLVServerSSLCertKeyBinding()

    @staticmethod
    def get(obj, servicegroup):
        if False:
            i = 10
            return i + 15
        '\n        Mock get method\n        '
        if MockNSNitro.flag:
            raise MockNSNitroError
        return [MockNSSSLVServerSSLCertKeyBinding()]

    @staticmethod
    def get_certkeyname():
        if False:
            return 10
        '\n        Mock get_certkeyname method\n        '
        return 'serviceGroupName'

    def set_certkeyname(self, sg_name):
        if False:
            while True:
                i = 10
        '\n        Mock set_certkeyname method\n        '
        self.sg_name = sg_name
        return MockNSSSLVServerSSLCertKeyBinding()

    @staticmethod
    def add(obj, servicegroup):
        if False:
            i = 10
            return i + 15
        '\n        Mock add method\n        '
        if MockNSNitro.flag:
            raise MockNSNitroError
        return MockNSSSLVServerSSLCertKeyBinding()

    @staticmethod
    def delete(obj, servicegroup):
        if False:
            return 10
        '\n        Mock delete method\n        '
        if MockNSNitro.flag:
            raise MockNSNitroError
        return MockNSSSLVServerSSLCertKeyBinding()

class NetscalerTestCase(TestCase, LoaderModuleMockMixin):
    """
    TestCase for salt.modules.netscaler
    """

    def setup_loader_modules(self):
        if False:
            i = 10
            return i + 15
        return {netscaler: {'NSNitro': MockNSNitro, 'NSServiceGroup': MockNSServiceGroup, 'NSServiceGroupServerBinding': MockNSServiceGroupServerBinding, 'NSLBVServerServiceGroupBinding': MockNSLBVServerServiceGroupBinding, 'NSService': MockNSService, 'NSServer': MockNSServer, 'NSLBVServer': MockNSLBVServer, 'NSNitroError': MockNSNitroError, 'NSSSLVServerSSLCertKeyBinding': MockNSSSLVServerSSLCertKeyBinding}}

    def test_servicegroup_exists(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests if it checks if a service group exists\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            MockNSNitro.flag = None
            self.assertTrue(netscaler.servicegroup_exists('serviceGrpName'))
            self.assertFalse(netscaler.servicegroup_exists('serviceGrpName', sg_type='HTTP'))
            MockNSNitro.flag = True
            with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                self.assertFalse(netscaler.servicegroup_exists('serGrpNme'))

    def test_servicegroup_add(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests if it add a new service group\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            self.assertFalse(netscaler.servicegroup_add('serviceGroupName'))
            MockNSNitro.flag = True
            self.assertFalse(netscaler.servicegroup_add('serviceGroupName'))
            with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                self.assertFalse(netscaler.servicegroup_add('serveGrpName'))

    def test_servicegroup_delete(self):
        if False:
            return 10
        '\n        Tests if it delete a new service group\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            MockNSNitro.flag = None
            self.assertTrue(netscaler.servicegroup_delete('serviceGrpName'))
            mock = MagicMock(side_effect=[None, MockNSServiceGroup()])
            with patch.object(netscaler, '_servicegroup_get', mock):
                MockNSNitro.flag = True
                self.assertFalse(netscaler.servicegroup_delete('srGrpName'))
                with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                    self.assertFalse(netscaler.servicegroup_delete('sGNam'))

    def test_servicegroup_server_exists(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests if it check if a server:port combination\n        is a member of a servicegroup\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            self.assertFalse(netscaler.servicegroup_server_exists('serviceGrpName', 'serverName', 'serverPort'))

    def test_servicegroup_server_up(self):
        if False:
            print('Hello World!')
        '\n        Tests if it check if a server:port combination\n        is a member of a servicegroup\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            self.assertFalse(netscaler.servicegroup_server_up('serviceGrpName', 'serverName', 'serverPort'))

    def test_servicegroup_server_enable(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests if it enable a server:port member of a servicegroup\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            self.assertFalse(netscaler.servicegroup_server_enable('serviceGrpName', 'serverName', 'serverPort'))
            with patch.object(netscaler, '_servicegroup_get_server', MagicMock(return_value=MockNSServiceGroup())):
                MockNSNitro.flag = None
                self.assertTrue(netscaler.servicegroup_server_enable('servGrpName', 'serverName', 'serPort'))
                with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                    self.assertFalse(netscaler.servicegroup_server_enable('serGrpName', 'serverName', 'sPort'))

    def test_sergrp_server_disable(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests if it disable a server:port member of a servicegroup\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            self.assertFalse(netscaler.servicegroup_server_disable('serviceGrpName', 'serverName', 'serverPort'))
            with patch.object(netscaler, '_servicegroup_get_server', MagicMock(return_value=MockNSServiceGroup())):
                MockNSNitro.flag = None
                self.assertTrue(netscaler.servicegroup_server_disable('serveGrpName', 'serverName', 'serPort'))
                with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                    self.assertFalse(netscaler.servicegroup_server_disable('servGrpName', 'serverName', 'sPort'))

    def test_servicegroup_server_add(self):
        if False:
            return 10
        '\n        Tests if it add a server:port member to a servicegroup\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                self.assertFalse(netscaler.servicegroup_server_add('serGrpName', 'serverName', 'sPort'))
            MockNSNitro.flag = None
            self.assertTrue(netscaler.servicegroup_server_add('serGrpName', 'serverName', 'serverPort'))
            mock = MagicMock(return_value=MockNSServiceGroupServerBinding())
            with patch.object(netscaler, '_servicegroup_get_server', mock):
                MockNSNitro.flag = True
                self.assertFalse(netscaler.servicegroup_server_add('serviceGroupName', 'serverName', 'serPort'))

    def test_servicegroup_server_delete(self):
        if False:
            while True:
                i = 10
        '\n        Tests if it remove a server:port member to a servicegroup\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                self.assertFalse(netscaler.servicegroup_server_delete('servGrpName', 'serverName', 'sPort'))
            self.assertFalse(netscaler.servicegroup_server_delete('serviceGroupName', 'serverName', 'serverPort'))
            mock = MagicMock(return_value=MockNSServiceGroupServerBinding())
            with patch.object(netscaler, '_servicegroup_get_server', mock):
                MockNSNitro.flag = None
                self.assertTrue(netscaler.servicegroup_server_delete('serviceGroupName', 'serverName', 'serPort'))

    def test_service_up(self):
        if False:
            print('Hello World!')
        '\n        Tests if it checks if a service is UP\n        '
        mock = MagicMock(return_value=MockNSService())
        with patch.object(netscaler, '_service_get', mock):
            self.assertTrue(netscaler.service_up('serviceGrpName'))

    def test_service_exists(self):
        if False:
            while True:
                i = 10
        '\n        Tests if it checks if a service is UP\n        '
        mock = MagicMock(return_value=MockNSService())
        with patch.object(netscaler, '_service_get', mock):
            self.assertTrue(netscaler.service_exists('serviceGrpName'))

    def test_service_enable(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests if it enable a service\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            self.assertTrue(netscaler.service_enable('serviceGrpName'))
            with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                self.assertFalse(netscaler.service_enable('serviceGrpName'))
                mock = MagicMock(return_value=MockNSService())
                with patch.object(netscaler, '_service_get', mock):
                    self.assertFalse(netscaler.service_enable('serGrpName'))

    def test_service_disable(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests if it disable a service\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            self.assertTrue(netscaler.service_disable('serviceGrpName'))
            with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                self.assertFalse(netscaler.service_disable('serceGrpName'))
                mock = MagicMock(return_value=MockNSService())
                with patch.object(netscaler, '_service_get', mock):
                    self.assertFalse(netscaler.service_disable('seGrpName'))

    def test_server_exists(self):
        if False:
            while True:
                i = 10
        '\n        Tests if it checks if a server exists\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            self.assertTrue(netscaler.server_exists('serviceGrpName'))
            with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                self.assertFalse(netscaler.server_exists('serviceGrpName'))
            self.assertFalse(netscaler.server_exists('serviceGrpName', ip='1.0.0.1'))
            self.assertFalse(netscaler.server_exists('serviceGrpName', s_state='serverName'))

    def test_server_add(self):
        if False:
            return 10
        '\n        Tests if it add a server\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            self.assertFalse(netscaler.server_add('servGrpName', '1.0.0.1'))
            with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                self.assertFalse(netscaler.server_add('serviceGrpName', '1.0.0.1'))
            mock = MagicMock(return_value=False)
            with patch.object(netscaler, 'server_exists', mock):
                self.assertTrue(netscaler.server_add('serviceGrpName', '1.0.0.1'))

    def test_server_delete(self):
        if False:
            print('Hello World!')
        '\n        Tests if it delete a server\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            self.assertTrue(netscaler.server_delete('serviceGrpName'))
            mock = MagicMock(side_effect=[MockNSServer(), None])
            with patch.object(netscaler, '_server_get', mock):
                with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                    self.assertFalse(netscaler.server_delete('serGrpName'))
                self.assertFalse(netscaler.server_delete('serviceGrpName'))

    def test_server_update(self):
        if False:
            while True:
                i = 10
        "\n        Tests if it update a server's attributes\n        "
        mock = MagicMock(side_effect=[None, MockNSServer(), MockNSServer(), MockNSServer()])
        with patch.object(netscaler, '_server_get', mock):
            self.assertFalse(netscaler.server_update('seGrName', '1.0.0.1'))
            self.assertFalse(netscaler.server_update('serGrpName', ''))
            with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                self.assertFalse(netscaler.server_update('serGrpName', '1.0.0.1'))
            mock = MagicMock(return_value='')
            with patch.dict(netscaler.__salt__, {'config.option': mock}):
                self.assertTrue(netscaler.server_update('serGrpName', '1.0.0.1'))

    def test_server_enabled(self):
        if False:
            print('Hello World!')
        '\n        Tests if it check if a server is enabled globally\n        '
        mock = MagicMock(return_value=MockNSServer())
        with patch.object(netscaler, '_server_get', mock):
            MockNSServer.flag = None
            self.assertTrue(netscaler.server_enabled('serGrpName'))

    def test_server_enable(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests if it enables a server globally\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            self.assertTrue(netscaler.server_enable('serGrpName'))
            MockNSServer.flag = 1
            self.assertTrue(netscaler.server_enable('serGrpName'))
            mock = MagicMock(side_effect=[MockNSServer(), None])
            with patch.object(netscaler, '_server_get', mock):
                with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                    self.assertFalse(netscaler.server_enable('serGrpName'))
                self.assertFalse(netscaler.server_enable('serGrpName'))

    def test_server_disable(self):
        if False:
            print('Hello World!')
        '\n        Tests if it disable a server globally\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            self.assertTrue(netscaler.server_disable('serGrpName'))
            MockNSServer.flag = 2
            self.assertTrue(netscaler.server_disable('serGrpName'))
            MockNSServer.flag = None
            mock = MagicMock(side_effect=[None, MockNSServer()])
            with patch.object(netscaler, '_server_get', mock):
                self.assertFalse(netscaler.server_disable('serGrpName'))
                with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                    self.assertFalse(netscaler.server_disable('serGrpName'))

    def test_vserver_exists(self):
        if False:
            return 10
        '\n        Tests if it checks if a vserver exists\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            self.assertTrue(netscaler.vserver_exists('vserverName'))
            self.assertFalse(netscaler.vserver_exists('vserverName', v_ip='1.0.0.1'))
            self.assertFalse(netscaler.vserver_exists('vserrName', v_ip='', v_port='vserverPort'))
            self.assertFalse(netscaler.vserver_exists('vserrName', v_ip='', v_port='', v_type='vserverType'))
            mock = MagicMock(return_value=None)
            with patch.object(netscaler, '_vserver_get', mock):
                self.assertFalse(netscaler.vserver_exists('vserverName'))

    def test_vserver_add(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests if it add a new lb vserver\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            self.assertFalse(netscaler.vserver_add('alex.patate.chaude.443', '1.2.3.4', '443', 'SSL'))
            mock = MagicMock(return_value=False)
            with patch.object(netscaler, 'vserver_exists', mock):
                self.assertTrue(netscaler.vserver_add('alex.pae.chaude.443', '1.2.3.4', '443', 'SSL'))
                with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                    self.assertFalse(netscaler.vserver_add('alex.chde.443', '1.2.3.4', '443', 'SSL'))

    def test_vserver_delete(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests if it delete a new lb vserver\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            self.assertTrue(netscaler.vserver_delete('alex.pe.chaude.443'))
            mock = MagicMock(side_effect=[None, MockNSLBVServer()])
            with patch.object(netscaler, '_vserver_get', mock):
                self.assertFalse(netscaler.vserver_delete('alex.chade.443'))
                with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                    self.assertFalse(netscaler.vserver_delete('al.cha.443'))

    def test_vser_sergrp_exists(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests if it checks if a servicegroup is tied to a vserver\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            self.assertTrue(netscaler.vserver_servicegroup_exists('vserverName', 'serviceGroupName'))

    def test_vserver_servicegroup_add(self):
        if False:
            return 10
        '\n        Tests if it bind a servicegroup to a vserver\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            MockNSNitro.flag = None
            self.assertTrue(netscaler.vserver_servicegroup_add('vserverName', 'serGroupName'))
            mock = MagicMock(side_effect=[MockNSLBVServerServiceGroupBinding(), None])
            with patch.object(netscaler, 'vserver_servicegroup_exists', mock):
                self.assertFalse(netscaler.vserver_servicegroup_add('vserName', 'serGroupName'))
                with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                    self.assertFalse(netscaler.vserver_servicegroup_add('vName', 'serGroupName'))

    def test_vser_sergrp_delete(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests if it unbind a servicegroup from a vserver\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            self.assertFalse(netscaler.vserver_servicegroup_delete('vservName', 'serGroupName'))
            mock = MagicMock(return_value=MockNSLBVServerServiceGroupBinding())
            with patch.object(netscaler, 'vserver_servicegroup_exists', mock):
                MockNSNitro.flag = None
                self.assertTrue(netscaler.vserver_servicegroup_delete('vName', 'serGroupName'))
                with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                    self.assertFalse(netscaler.vserver_servicegroup_delete('vserverName', 'serGroupName'))

    def test_vserver_sslcert_exists(self):
        if False:
            return 10
        '\n        Tests if it checks if a SSL certificate is tied to a vserver\n        '
        mock = MagicMock(return_value='')
        with patch.dict(netscaler.__salt__, {'config.option': mock}):
            self.assertTrue(netscaler.vserver_sslcert_exists('vserverName', 'serviceGroupName'))

    def test_vserver_sslcert_add(self):
        if False:
            i = 10
            return i + 15
        '\n        Tests if it binds a SSL certificate to a vserver\n        '
        mock = MagicMock(side_effect=[MockNSSSLVServerSSLCertKeyBinding(), None, None])
        with patch.object(netscaler, 'vserver_sslcert_exists', mock):
            self.assertFalse(netscaler.vserver_sslcert_add('vserName', 'serGroupName'))
            with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                self.assertFalse(netscaler.vserver_sslcert_add('vName', 'serGrName'))
            mock = MagicMock(return_value='')
            with patch.dict(netscaler.__salt__, {'config.option': mock}):
                MockNSNitro.flag = None
                self.assertTrue(netscaler.vserver_sslcert_add('vserverName', 'serGroupName'))

    def test_vserver_sslcert_delete(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tests if it unbinds a SSL certificate from a vserver\n        '
        mock = MagicMock(side_effect=[None, MockNSSSLVServerSSLCertKeyBinding(), MockNSSSLVServerSSLCertKeyBinding()])
        with patch.object(netscaler, 'vserver_sslcert_exists', mock):
            self.assertFalse(netscaler.vserver_sslcert_delete('vName', 'serGrpName'))
            mock = MagicMock(return_value='')
            with patch.dict(netscaler.__salt__, {'config.option': mock}):
                MockNSNitro.flag = None
                self.assertTrue(netscaler.vserver_sslcert_delete('vservName', 'serGroupName'))
            with patch.object(netscaler, '_connect', MagicMock(return_value=None)):
                self.assertFalse(netscaler.vserver_sslcert_delete('vserverName', 'serGroupName'))