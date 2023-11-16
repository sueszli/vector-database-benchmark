import asyncio
import logging
import pytest
import tornado.gen
import tornado.iostream
import tornado.tcpserver
import salt.transport.tcp
import salt.utils.msgpack
log = logging.getLogger(__name__)

@pytest.fixture
def config():
    if False:
        while True:
            i = 10
    yield {'master_ip': '127.0.0.1', 'publish_port': 5679}

@pytest.fixture
def server(config):
    if False:
        while True:
            i = 10

    class TestServer(tornado.tcpserver.TCPServer):
        send = []
        disconnect = False

        async def handle_stream(self, stream, address):
            try:
                log.info('Got stream %r', self.disconnect)
                while self.disconnect is False:
                    for msg in self.send[:]:
                        msg = self.send.pop(0)
                        try:
                            log.info('Write %r', msg)
                            await stream.write(msg)
                        except tornado.iostream.StreamClosedError:
                            log.error('Stream Closed Error From Test Server')
                            break
                    else:
                        log.info('Sleep')
                        await asyncio.sleep(1)
                log.info('Close stream')
            finally:
                stream.close()
                log.info('After close stream')
    server = TestServer()
    try:
        yield server
    finally:
        server.disconnect = True
        server.stop()

@pytest.fixture
def client(io_loop, config):
    if False:
        return 10
    client = salt.transport.tcp.TCPPubClient(config.copy(), io_loop, host=config['master_ip'], port=config['publish_port'])
    try:
        yield client
    finally:
        client.close()

async def test_message_client_reconnect(config, client, server):
    """
    Verify that the tcp MessageClient class re-sets it's unpacker after a
    stream disconnect.
    """
    server.listen(config['publish_port'])
    await client.connect(config['publish_port'])
    received = []

    def handler(msg):
        if False:
            i = 10
            return i + 15
        received.append(msg)
    client.on_recv(handler)
    msg = salt.utils.msgpack.dumps({'test': 'test1'})
    pmsg = salt.utils.msgpack.dumps({'head': {}, 'body': msg})
    assert len(pmsg) == 26
    pmsg += salt.utils.msgpack.dumps({'head': {}, 'body': msg})
    partial = pmsg[:40]
    log.info('Send partial %r', partial)
    server.send.append(partial)
    while not received:
        log.info('wait received')
        await asyncio.sleep(1)
    log.info('assert received')
    assert received == [msg]
    server.disconnect = True
    await asyncio.sleep(1)
    server.disconnect = False
    await asyncio.sleep(1)
    received = []
    server.send.append(pmsg)
    while not received:
        await tornado.gen.sleep(1)
    assert received == [msg, msg]
    server.disconnect = True
    client.close()
    await tornado.gen.sleep(1)