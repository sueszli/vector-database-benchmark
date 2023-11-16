import logging
import os
from typing import Any, Optional, Tuple
from twisted.internet.protocol import Factory
from twisted.test.proto_helpers import MemoryReactor
from twisted.web.http import HTTPChannel
from twisted.web.server import Request
from synapse.rest import admin
from synapse.rest.client import login
from synapse.server import HomeServer
from synapse.util import Clock
from tests.http import TestServerTLSConnectionFactory, get_test_ca_cert_file, wrap_server_factory_for_tls
from tests.replication._base import BaseMultiWorkerStreamTestCase
from tests.server import FakeChannel, FakeTransport, make_request
from tests.test_utils import SMALL_PNG
logger = logging.getLogger(__name__)
test_server_connection_factory: Optional[TestServerTLSConnectionFactory] = None

class MediaRepoShardTestCase(BaseMultiWorkerStreamTestCase):
    """Checks running multiple media repos work correctly."""
    servlets = [admin.register_servlets_for_client_rest_resource, login.register_servlets]

    def prepare(self, reactor: MemoryReactor, clock: Clock, hs: HomeServer) -> None:
        if False:
            print('Hello World!')
        self.user_id = self.register_user('user', 'pass')
        self.access_token = self.login('user', 'pass')
        self.reactor.lookups['example.com'] = '1.2.3.4'

    def default_config(self) -> dict:
        if False:
            i = 10
            return i + 15
        conf = super().default_config()
        conf['federation_custom_ca_list'] = [get_test_ca_cert_file()]
        return conf

    def make_worker_hs(self, worker_app: str, extra_config: Optional[dict]=None, **kwargs: Any) -> HomeServer:
        if False:
            while True:
                i = 10
        worker_hs = super().make_worker_hs(worker_app, extra_config, **kwargs)
        worker_hs.get_media_repository_resource().register_servlets(self._hs_to_site[worker_hs].resource, worker_hs)
        return worker_hs

    def _get_media_req(self, hs: HomeServer, target: str, media_id: str) -> Tuple[FakeChannel, Request]:
        if False:
            return 10
        'Request some remote media from the given HS by calling the download\n        API.\n\n        This then triggers an outbound request from the HS to the target.\n\n        Returns:\n            The channel for the *client* request and the *outbound* request for\n            the media which the caller should respond to.\n        '
        channel = make_request(self.reactor, self._hs_to_site[hs], 'GET', f'/_matrix/media/r0/download/{target}/{media_id}', shorthand=False, access_token=self.access_token, await_result=False)
        self.pump()
        clients = self.reactor.tcpClients
        self.assertGreaterEqual(len(clients), 1)
        (host, port, client_factory, _timeout, _bindAddress) = clients.pop()
        server_factory = Factory.forProtocol(HTTPChannel)
        server_factory.log = _log_request
        server_tls_protocol = wrap_server_factory_for_tls(server_factory, self.reactor, sanlist=[b'DNS:example.com']).buildProtocol(None)
        client_protocol = client_factory.buildProtocol(None)
        client_protocol.makeConnection(FakeTransport(server_tls_protocol, self.reactor, client_protocol))
        server_tls_protocol.makeConnection(FakeTransport(client_protocol, self.reactor, server_tls_protocol))
        http_server: HTTPChannel = server_tls_protocol.wrappedProtocol
        self.reactor.pump((0.1,))
        self.assertEqual(len(http_server.requests), 1)
        request = http_server.requests[0]
        self.assertEqual(request.method, b'GET')
        self.assertEqual(request.path, f'/_matrix/media/r0/download/{target}/{media_id}'.encode())
        self.assertEqual(request.requestHeaders.getRawHeaders(b'host'), [target.encode('utf-8')])
        return (channel, request)

    def test_basic(self) -> None:
        if False:
            return 10
        'Test basic fetching of remote media from a single worker.'
        hs1 = self.make_worker_hs('synapse.app.generic_worker')
        (channel, request) = self._get_media_req(hs1, 'example.com:443', 'ABC123')
        request.setResponseCode(200)
        request.responseHeaders.setRawHeaders(b'Content-Type', [b'text/plain'])
        request.write(b'Hello!')
        request.finish()
        self.pump(0.1)
        self.assertEqual(channel.code, 200)
        self.assertEqual(channel.result['body'], b'Hello!')

    def test_download_simple_file_race(self) -> None:
        if False:
            while True:
                i = 10
        'Test that fetching remote media from two different processes at the\n        same time works.\n        '
        hs1 = self.make_worker_hs('synapse.app.generic_worker')
        hs2 = self.make_worker_hs('synapse.app.generic_worker')
        start_count = self._count_remote_media()
        (channel1, request1) = self._get_media_req(hs1, 'example.com:443', 'ABC123')
        (channel2, request2) = self._get_media_req(hs2, 'example.com:443', 'ABC123')
        request1.setResponseCode(200)
        request1.responseHeaders.setRawHeaders(b'Content-Type', [b'text/plain'])
        request1.write(b'Hello!')
        request1.finish()
        self.pump(0.1)
        self.assertEqual(channel1.code, 200, channel1.result['body'])
        self.assertEqual(channel1.result['body'], b'Hello!')
        request2.setResponseCode(200)
        request2.responseHeaders.setRawHeaders(b'Content-Type', [b'text/plain'])
        request2.write(b'Hello!')
        request2.finish()
        self.pump(0.1)
        self.assertEqual(channel2.code, 200, channel2.result['body'])
        self.assertEqual(channel2.result['body'], b'Hello!')
        self.assertEqual(start_count + 1, self._count_remote_media())

    def test_download_image_race(self) -> None:
        if False:
            i = 10
            return i + 15
        'Test that fetching remote *images* from two different processes at\n        the same time works.\n\n        This checks that races generating thumbnails are handled correctly.\n        '
        hs1 = self.make_worker_hs('synapse.app.generic_worker')
        hs2 = self.make_worker_hs('synapse.app.generic_worker')
        start_count = self._count_remote_thumbnails()
        (channel1, request1) = self._get_media_req(hs1, 'example.com:443', 'PIC1')
        (channel2, request2) = self._get_media_req(hs2, 'example.com:443', 'PIC1')
        request1.setResponseCode(200)
        request1.responseHeaders.setRawHeaders(b'Content-Type', [b'image/png'])
        request1.write(SMALL_PNG)
        request1.finish()
        self.pump(0.1)
        self.assertEqual(channel1.code, 200, channel1.result['body'])
        self.assertEqual(channel1.result['body'], SMALL_PNG)
        request2.setResponseCode(200)
        request2.responseHeaders.setRawHeaders(b'Content-Type', [b'image/png'])
        request2.write(SMALL_PNG)
        request2.finish()
        self.pump(0.1)
        self.assertEqual(channel2.code, 200, channel2.result['body'])
        self.assertEqual(channel2.result['body'], SMALL_PNG)
        self.assertEqual(start_count + 3, self._count_remote_thumbnails())

    def _count_remote_media(self) -> int:
        if False:
            return 10
        'Count the number of files in our remote media directory.'
        path = os.path.join(self.hs.get_media_repository().primary_base_path, 'remote_content')
        return sum((len(files) for (_, _, files) in os.walk(path)))

    def _count_remote_thumbnails(self) -> int:
        if False:
            i = 10
            return i + 15
        'Count the number of files in our remote thumbnails directory.'
        path = os.path.join(self.hs.get_media_repository().primary_base_path, 'remote_thumbnail')
        return sum((len(files) for (_, _, files) in os.walk(path)))

def _log_request(request: Request) -> None:
    if False:
        print('Hello World!')
    'Implements Factory.log, which is expected by Request.finish'
    logger.info('Completed request %s', request)