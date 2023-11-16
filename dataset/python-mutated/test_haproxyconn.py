"""
    :codeauthor: Jayesh Kariya <jayeshk@saltstack.com>

    Test cases for salt.modules.haproxyconn
"""
import pytest
import salt.modules.haproxyconn as haproxyconn

class Mockcmds:
    """
    Mock of cmds
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.backend = None
        self.server = None
        self.weight = None

    def listServers(self, backend):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock of listServers method\n        '
        self.backend = backend
        return 'Name: server01 Status: UP Weight: 1 bIn: 22 bOut: 12\nName: server02 Status: MAINT Weight: 2 bIn: 0 bOut: 0'

    def enableServer(self, server, backend):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mock of enableServer method\n        '
        self.backend = backend
        self.server = server
        return 'server enabled'

    def disableServer(self, server, backend):
        if False:
            i = 10
            return i + 15
        '\n        Mock of disableServer method\n        '
        self.backend = backend
        self.server = server
        return 'server disabled'

    def getWeight(self, server, backend, weight=0):
        if False:
            return 10
        '\n        Mock of getWeight method\n        '
        self.backend = backend
        self.server = server
        self.weight = weight
        return 'server weight'

    @staticmethod
    def showFrontends():
        if False:
            while True:
                i = 10
        '\n        Mock of showFrontends method\n        '
        return 'frontend-alpha\nfrontend-beta\nfrontend-gamma'

    @staticmethod
    def showBackends():
        if False:
            while True:
                i = 10
        '\n        Mock of showBackends method\n        '
        return 'backend-alpha\nbackend-beta\nbackend-gamma'

class Mockhaproxy:
    """
    Mock of haproxy
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.cmds = Mockcmds()

class MockHaConn:
    """
    Mock of HaConn
    """

    def __init__(self, socket=None):
        if False:
            return 10
        self.ha_cmd = None

    def sendCmd(self, ha_cmd, objectify=False):
        if False:
            print('Hello World!')
        '\n        Mock of sendCmd method\n        '
        self.ha_cmd = ha_cmd
        self.objectify = objectify
        return ha_cmd

@pytest.fixture
def configure_loader_modules():
    if False:
        return 10
    return {haproxyconn: {'haproxy': Mockhaproxy(), '_get_conn': MockHaConn}}

def test_list_servers():
    if False:
        print('Hello World!')
    '\n    Test list_servers\n    '
    assert haproxyconn.list_servers('mysql')

def test_enable_server():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test enable_server\n    '
    assert haproxyconn.enable_server('web1.salt.com', 'www')

def test_disable_server():
    if False:
        return 10
    '\n    Test disable_server\n    '
    assert haproxyconn.disable_server('db1.salt.com', 'mysql')

def test_get_weight():
    if False:
        print('Hello World!')
    '\n    Test get the weight of a server\n    '
    assert haproxyconn.get_weight('db1.salt.com', 'mysql')

def test_set_weight():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test setting the weight of a given server\n    '
    assert haproxyconn.set_weight('db1.salt.com', 'mysql', weight=11)

def test_show_frontends():
    if False:
        print('Hello World!')
    '\n    Test print all frontends received from the HAProxy socket\n    '
    assert haproxyconn.show_frontends()

def test_list_frontends():
    if False:
        print('Hello World!')
    '\n    Test listing all frontends\n    '
    assert sorted(haproxyconn.list_frontends()) == sorted(['frontend-alpha', 'frontend-beta', 'frontend-gamma'])

def test_show_backends():
    if False:
        print('Hello World!')
    '\n    Test print all backends received from the HAProxy socket\n    '
    assert haproxyconn.show_backends()

def test_list_backends():
    if False:
        print('Hello World!')
    '\n    Test listing of all backends\n    '
    assert sorted(haproxyconn.list_backends()) == sorted(['backend-alpha', 'backend-beta', 'backend-gamma'])

def test_get_backend():
    if False:
        while True:
            i = 10
    '\n    Test get_backend and compare returned value\n    '
    expected_data = {'server01': {'status': 'UP', 'weight': 1, 'bin': 22, 'bout': 12}, 'server02': {'status': 'MAINT', 'weight': 2, 'bin': 0, 'bout': 0}}
    assert haproxyconn.get_backend('test') == expected_data

def test_wait_state_true():
    if False:
        print('Hello World!')
    '\n    Test a successful wait for state\n    '
    assert haproxyconn.wait_state('test', 'server01')

def test_wait_state_false():
    if False:
        return 10
    '\n    Test a failed wait for state, with a timeout of 0\n    '
    assert not haproxyconn.wait_state('test', 'server02', 'up', 0)