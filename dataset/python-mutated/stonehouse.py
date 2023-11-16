"""
Stonehouse uses the "CURVE" security mechanism.

This gives us strong encryption on data, and (as far as we know) unbreakable
authentication. Stonehouse is the minimum you would use over public networks,
and assures clients that they are speaking to an authentic server, while
allowing any client to connect.

Author: Chris Laws
"""
import logging
import os
import sys
import zmq
import zmq.auth
from zmq.auth.thread import ThreadAuthenticator

def run() -> None:
    if False:
        i = 10
        return i + 15
    'Run Stonehouse example'
    base_dir = os.path.dirname(__file__)
    keys_dir = os.path.join(base_dir, 'certificates')
    public_keys_dir = os.path.join(base_dir, 'public_keys')
    secret_keys_dir = os.path.join(base_dir, 'private_keys')
    if not (os.path.exists(keys_dir) and os.path.exists(public_keys_dir) and os.path.exists(secret_keys_dir)):
        logging.critical('Certificates are missing: run generate_certificates.py script first')
        sys.exit(1)
    ctx = zmq.Context.instance()
    auth = ThreadAuthenticator(ctx)
    auth.start()
    auth.allow('127.0.0.1')
    auth.configure_curve(domain='*', location=zmq.auth.CURVE_ALLOW_ANY)
    server = ctx.socket(zmq.PUSH)
    server_secret_file = os.path.join(secret_keys_dir, 'server.key_secret')
    (server_public, server_secret) = zmq.auth.load_certificate(server_secret_file)
    server.curve_secretkey = server_secret
    server.curve_publickey = server_public
    server.curve_server = True
    server.bind('tcp://*:9000')
    client = ctx.socket(zmq.PULL)
    client_secret_file = os.path.join(secret_keys_dir, 'client.key_secret')
    (client_public, client_secret) = zmq.auth.load_certificate(client_secret_file)
    client.curve_secretkey = client_secret
    client.curve_publickey = client_public
    server_public_file = os.path.join(public_keys_dir, 'server.key')
    (server_public, _) = zmq.auth.load_certificate(server_public_file)
    client.curve_serverkey = server_public
    client.connect('tcp://127.0.0.1:9000')
    server.send(b'Hello')
    if client.poll(1000):
        msg = client.recv()
        if msg == b'Hello':
            logging.info('Stonehouse test OK')
    else:
        logging.error('Stonehouse test FAIL')
    auth.stop()
if __name__ == '__main__':
    if zmq.zmq_version_info() < (4, 0):
        raise RuntimeError('Security is not supported in libzmq version < 4.0. libzmq version {}'.format(zmq.zmq_version()))
    if '-v' in sys.argv:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')
    run()