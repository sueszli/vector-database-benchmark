import time
import socket
import gevent
import pytest
import mock
from Crypt import CryptConnection
from Connection import ConnectionServer
from Config import config

@pytest.mark.usefixtures('resetSettings')
class TestConnection:

    def testIpv6(self, file_server6):
        if False:
            while True:
                i = 10
        assert ':' in file_server6.ip
        client = ConnectionServer(file_server6.ip, 1545)
        connection = client.getConnection(file_server6.ip, 1544)
        assert connection.ping()
        connection.close()
        client.stop()
        time.sleep(0.01)
        assert len(file_server6.connections) == 0
        with pytest.raises(socket.error) as err:
            client = ConnectionServer('127.0.0.1', 1545)
            connection = client.getConnection('127.0.0.1', 1544)

    def testSslConnection(self, file_server):
        if False:
            i = 10
            return i + 15
        client = ConnectionServer(file_server.ip, 1545)
        assert file_server != client
        with mock.patch('Config.config.ip_local', return_value=[]):
            connection = client.getConnection(file_server.ip, 1544)
        assert len(file_server.connections) == 1
        assert connection.handshake
        assert connection.crypt
        connection.close('Test ended')
        client.stop()
        time.sleep(0.1)
        assert len(file_server.connections) == 0
        assert file_server.num_incoming == 2

    def testRawConnection(self, file_server):
        if False:
            for i in range(10):
                print('nop')
        client = ConnectionServer(file_server.ip, 1545)
        assert file_server != client
        crypt_supported_bk = CryptConnection.manager.crypt_supported
        CryptConnection.manager.crypt_supported = []
        with mock.patch('Config.config.ip_local', return_value=[]):
            connection = client.getConnection(file_server.ip, 1544)
        assert len(file_server.connections) == 1
        assert not connection.crypt
        connection.close()
        client.stop()
        time.sleep(0.01)
        assert len(file_server.connections) == 0
        CryptConnection.manager.crypt_supported = crypt_supported_bk

    def testPing(self, file_server, site):
        if False:
            print('Hello World!')
        client = ConnectionServer(file_server.ip, 1545)
        connection = client.getConnection(file_server.ip, 1544)
        assert connection.ping()
        connection.close()
        client.stop()

    def testGetConnection(self, file_server):
        if False:
            print('Hello World!')
        client = ConnectionServer(file_server.ip, 1545)
        connection = client.getConnection(file_server.ip, 1544)
        connection2 = client.getConnection(file_server.ip, 1544)
        assert connection == connection2
        assert not client.getConnection(file_server.ip, 1544, peer_id='notexists', create=False)
        connection2 = client.getConnection(file_server.ip, 1544, peer_id=connection.handshake['peer_id'], create=False)
        assert connection2 == connection
        connection.close()
        client.stop()

    def testFloodProtection(self, file_server):
        if False:
            i = 10
            return i + 15
        whitelist = file_server.whitelist
        file_server.whitelist = []
        client = ConnectionServer(file_server.ip, 1545)
        for reconnect in range(6):
            connection = client.getConnection(file_server.ip, 1544)
            assert connection.handshake
            connection.close()
        with pytest.raises(gevent.Timeout):
            with gevent.Timeout(0.1):
                connection = client.getConnection(file_server.ip, 1544)
        file_server.whitelist = whitelist