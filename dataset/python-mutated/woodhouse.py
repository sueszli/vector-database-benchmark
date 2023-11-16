"""
Woodhouse extends Strawhouse with a name and password check.

This uses the PLAIN mechanism which does plain-text username and password authentication).
It's not really secure, and anyone sniffing the network (trivial with WiFi)
can capture passwords and then login.

Author: Chris Laws
"""
import logging
import sys
import zmq
import zmq.auth
from zmq.auth.thread import ThreadAuthenticator

def run() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Run woodhouse example'
    valid_client_test_pass = False
    invalid_client_test_pass = False
    ctx = zmq.Context.instance()
    auth = ThreadAuthenticator(ctx)
    auth.start()
    auth.allow('127.0.0.1')
    auth.configure_plain(domain='*', passwords={'admin': 'secret'})
    server = ctx.socket(zmq.PUSH)
    server.plain_server = True
    server.bind('tcp://*:9000')
    client = ctx.socket(zmq.PULL)
    client.plain_username = b'admin'
    client.plain_password = b'secret'
    client.connect('tcp://127.0.0.1:9000')
    server.send(b'Hello')
    if client.poll():
        msg = client.recv()
        if msg == b'Hello':
            valid_client_test_pass = True
    client.close()
    client2 = ctx.socket(zmq.PULL)
    client2.plain_username = b'admin'
    client2.plain_password = b'bogus'
    client2.connect('tcp://127.0.0.1:9000')
    server.send(b'World')
    if client2.poll(50):
        msg = client.recv()
        if msg == 'World':
            invalid_client_test_pass = False
    else:
        invalid_client_test_pass = True
    auth.stop()
    if valid_client_test_pass and invalid_client_test_pass:
        logging.info('Woodhouse test OK')
    else:
        logging.error('Woodhouse test FAIL')
if __name__ == '__main__':
    if zmq.zmq_version_info() < (4, 0):
        raise RuntimeError('Security is not supported in libzmq version < 4.0. libzmq version {}'.format(zmq.zmq_version()))
    if '-v' in sys.argv:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')
    run()