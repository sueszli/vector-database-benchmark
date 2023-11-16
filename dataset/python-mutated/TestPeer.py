import time
import io
import pytest
from File import FileServer
from File import FileRequest
from Crypt import CryptHash
from . import Spy

@pytest.mark.usefixtures('resetSettings')
@pytest.mark.usefixtures('resetTempSettings')
class TestPeer:

    def testPing(self, file_server, site, site_temp):
        if False:
            return 10
        file_server.sites[site.address] = site
        client = FileServer(file_server.ip, 1545)
        client.sites = {site_temp.address: site_temp}
        site_temp.connection_server = client
        connection = client.getConnection(file_server.ip, 1544)
        peer_file_server = site_temp.addPeer(file_server.ip, 1544)
        assert peer_file_server.ping() is not None
        assert peer_file_server in site_temp.peers.values()
        peer_file_server.remove()
        assert peer_file_server not in site_temp.peers.values()
        connection.close()
        client.stop()

    def testDownloadFile(self, file_server, site, site_temp):
        if False:
            for i in range(10):
                print('nop')
        file_server.sites[site.address] = site
        client = FileServer(file_server.ip, 1545)
        client.sites = {site_temp.address: site_temp}
        site_temp.connection_server = client
        connection = client.getConnection(file_server.ip, 1544)
        peer_file_server = site_temp.addPeer(file_server.ip, 1544)
        buff = peer_file_server.getFile(site_temp.address, 'content.json', streaming=True)
        assert b'sign' in buff.getvalue()
        buff = peer_file_server.getFile(site_temp.address, 'content.json')
        assert b'sign' in buff.getvalue()
        connection.close()
        client.stop()

    def testHashfield(self, site):
        if False:
            while True:
                i = 10
        sample_hash = list(site.content_manager.contents['content.json']['files_optional'].values())[0]['sha512']
        site.storage.verifyFiles(quick_check=True)
        assert site.content_manager.hashfield
        assert len(site.content_manager.hashfield) > 0
        assert site.content_manager.hashfield.getHashId(sample_hash) in site.content_manager.hashfield
        new_hash = CryptHash.sha512sum(io.BytesIO(b'hello'))
        assert site.content_manager.hashfield.getHashId(new_hash) not in site.content_manager.hashfield
        assert site.content_manager.hashfield.appendHash(new_hash)
        assert not site.content_manager.hashfield.appendHash(new_hash)
        assert site.content_manager.hashfield.getHashId(new_hash) in site.content_manager.hashfield
        assert site.content_manager.hashfield.removeHash(new_hash)
        assert site.content_manager.hashfield.getHashId(new_hash) not in site.content_manager.hashfield

    def testHashfieldExchange(self, file_server, site, site_temp):
        if False:
            i = 10
            return i + 15
        server1 = file_server
        server1.sites[site.address] = site
        site.connection_server = server1
        server2 = FileServer(file_server.ip, 1545)
        server2.sites[site_temp.address] = site_temp
        site_temp.connection_server = server2
        site.storage.verifyFiles(quick_check=True)
        server2_peer1 = site_temp.addPeer(file_server.ip, 1544)
        assert len(site.content_manager.hashfield) > 0
        assert len(server2_peer1.hashfield) == 0
        assert server2_peer1.updateHashfield()
        assert len(server2_peer1.hashfield) > 0
        site_temp.content_manager.hashfield.appendHash('AABB')
        server1_peer2 = site.addPeer(file_server.ip, 1545, return_peer=True)
        with Spy.Spy(FileRequest, 'route') as requests:
            assert len(server1_peer2.hashfield) == 0
            server2_peer1.sendMyHashfield()
            assert len(server1_peer2.hashfield) == 1
            server2_peer1.sendMyHashfield()
            assert len(requests) == 1
            time.sleep(0.01)
            site_temp.content_manager.hashfield.appendHash('AACC')
            server2_peer1.sendMyHashfield()
            assert len(server1_peer2.hashfield) == 2
            assert len(requests) == 2
            site_temp.content_manager.hashfield.appendHash('AADD')
            assert server1_peer2.updateHashfield(force=True)
            assert len(server1_peer2.hashfield) == 3
            assert len(requests) == 3
            assert not server2_peer1.sendMyHashfield()
            assert len(requests) == 3
        server2.stop()

    def testFindHash(self, file_server, site, site_temp):
        if False:
            print('Hello World!')
        file_server.sites[site.address] = site
        client = FileServer(file_server.ip, 1545)
        client.sites = {site_temp.address: site_temp}
        site_temp.connection_server = client
        peer_file_server = site_temp.addPeer(file_server.ip, 1544)
        assert peer_file_server.findHashIds([1234]) == {}
        fake_peer_1 = site.addPeer(file_server.ip_external, 1544)
        fake_peer_1.hashfield.append(1234)
        fake_peer_2 = site.addPeer('1.2.3.5', 1545)
        fake_peer_2.hashfield.append(1234)
        fake_peer_2.hashfield.append(1235)
        fake_peer_3 = site.addPeer('1.2.3.6', 1546)
        fake_peer_3.hashfield.append(1235)
        fake_peer_3.hashfield.append(1236)
        res = peer_file_server.findHashIds([1234, 1235])
        assert sorted(res[1234]) == sorted([(file_server.ip_external, 1544), ('1.2.3.5', 1545)])
        assert sorted(res[1235]) == sorted([('1.2.3.5', 1545), ('1.2.3.6', 1546)])
        site.content_manager.hashfield.append(1234)
        res = peer_file_server.findHashIds([1234, 1235])
        assert sorted(res[1234]) == sorted([(file_server.ip_external, 1544), ('1.2.3.5', 1545), (file_server.ip, 1544)])
        assert sorted(res[1235]) == sorted([('1.2.3.5', 1545), ('1.2.3.6', 1546)])