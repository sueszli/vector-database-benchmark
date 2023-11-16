import time
import pytest
import mock
from File import FileServer
from Crypt import CryptRsa
from Config import config

@pytest.mark.usefixtures('resetSettings')
@pytest.mark.usefixtures('resetTempSettings')
class TestTor:

    def testDownload(self, tor_manager):
        if False:
            for i in range(10):
                print('nop')
        for retry in range(15):
            time.sleep(1)
            if tor_manager.enabled and tor_manager.conn:
                break
        assert tor_manager.enabled

    def testManagerConnection(self, tor_manager):
        if False:
            while True:
                i = 10
        assert '250-version' in tor_manager.request('GETINFO version')

    def testAddOnion(self, tor_manager):
        if False:
            print('Hello World!')
        address = tor_manager.addOnion()
        assert address
        assert address in tor_manager.privatekeys
        assert tor_manager.delOnion(address)
        assert address not in tor_manager.privatekeys

    def testSignOnion(self, tor_manager):
        if False:
            while True:
                i = 10
        address = tor_manager.addOnion()
        sign = CryptRsa.sign(b'hello', tor_manager.getPrivatekey(address))
        assert len(sign) == 128
        publickey = CryptRsa.privatekeyToPublickey(tor_manager.getPrivatekey(address))
        assert len(publickey) == 140
        assert CryptRsa.verify(b'hello', publickey, sign)
        assert not CryptRsa.verify(b'not hello', publickey, sign)
        assert CryptRsa.publickeyToOnion(publickey) == address
        tor_manager.delOnion(address)

    @pytest.mark.slow
    def testConnection(self, tor_manager, file_server, site, site_temp):
        if False:
            while True:
                i = 10
        file_server.tor_manager.start_onions = True
        address = file_server.tor_manager.getOnion(site.address)
        assert address
        print('Connecting to', address)
        for retry in range(5):
            time.sleep(10)
            try:
                connection = file_server.getConnection(address + '.onion', 1544)
                if connection:
                    break
            except Exception as err:
                continue
        assert connection.handshake
        assert not connection.handshake['peer_id']
        assert file_server.getConnection(address + '.onion', 1544) == connection
        assert file_server.getConnection(address + '.onion', 1544, site=site) != connection
        assert file_server.getConnection(address + '.onion', 1544, site=site) == file_server.getConnection(address + '.onion', 1544, site=site)
        site_temp.address = '1OTHERSITE'
        assert file_server.getConnection(address + '.onion', 1544, site=site) != file_server.getConnection(address + '.onion', 1544, site=site_temp)
        file_server.sites[site.address] = site
        connection_locked = file_server.getConnection(address + '.onion', 1544, site=site)
        assert 'body' in connection_locked.request('getFile', {'site': site.address, 'inner_path': 'content.json', 'location': 0})
        assert connection_locked.request('getFile', {'site': '1OTHERSITE', 'inner_path': 'content.json', 'location': 0})['error'] == 'Invalid site'

    def testPex(self, file_server, site, site_temp):
        if False:
            return 10
        site.connection_server = file_server
        file_server.sites[site.address] = site
        file_server_temp = FileServer(file_server.ip, 1545)
        site_temp.connection_server = file_server_temp
        file_server_temp.sites[site_temp.address] = site_temp
        peer_source = site_temp.addPeer(file_server.ip, 1544)
        site.addPeer('1.2.3.4', 1555)
        assert peer_source.pex(need_num=10) == 1
        assert len(site_temp.peers) == 2
        assert '1.2.3.4:1555' in site_temp.peers
        site.addPeer('bka4ht2bzxchy44r.onion', 1555)
        assert 'bka4ht2bzxchy44r.onion:1555' not in site_temp.peers
        assert 'onion' not in file_server_temp.supported_ip_types
        assert peer_source.pex(need_num=10) == 0
        file_server_temp.supported_ip_types.append('onion')
        assert peer_source.pex(need_num=10) == 1
        assert 'bka4ht2bzxchy44r.onion:1555' in site_temp.peers

    def testFindHash(self, tor_manager, file_server, site, site_temp):
        if False:
            print('Hello World!')
        file_server.ip_incoming = {}
        file_server.sites[site.address] = site
        file_server.tor_manager = tor_manager
        client = FileServer(file_server.ip, 1545)
        client.sites = {site_temp.address: site_temp}
        site_temp.connection_server = client
        peer_file_server = site_temp.addPeer(file_server.ip, 1544)
        assert peer_file_server.findHashIds([1234]) == {}
        fake_peer_1 = site.addPeer('bka4ht2bzxchy44r.onion', 1544)
        fake_peer_1.hashfield.append(1234)
        fake_peer_2 = site.addPeer('1.2.3.5', 1545)
        fake_peer_2.hashfield.append(1234)
        fake_peer_2.hashfield.append(1235)
        fake_peer_3 = site.addPeer('1.2.3.6', 1546)
        fake_peer_3.hashfield.append(1235)
        fake_peer_3.hashfield.append(1236)
        res = peer_file_server.findHashIds([1234, 1235])
        assert sorted(res[1234]) == [('1.2.3.5', 1545), ('bka4ht2bzxchy44r.onion', 1544)]
        assert sorted(res[1235]) == [('1.2.3.5', 1545), ('1.2.3.6', 1546)]
        site.content_manager.hashfield.append(1234)
        res = peer_file_server.findHashIds([1234, 1235])
        assert sorted(res[1234]) == [('1.2.3.5', 1545), (file_server.ip, 1544), ('bka4ht2bzxchy44r.onion', 1544)]
        assert sorted(res[1235]) == [('1.2.3.5', 1545), ('1.2.3.6', 1546)]

    def testSiteOnion(self, tor_manager):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch.object(config, 'tor', 'always'):
            assert tor_manager.getOnion('address1') != tor_manager.getOnion('address2')
            assert tor_manager.getOnion('address1') == tor_manager.getOnion('address1')