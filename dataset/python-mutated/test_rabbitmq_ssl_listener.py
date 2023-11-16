from __future__ import absolute_import
import os
import ssl
import socket
import six
import unittest2
from oslo_config import cfg
from st2common.transport import utils as transport_utils
from st2tests.fixtures.ssl_certs.fixture import FIXTURE_PATH as CERTS_FIXTURES_PATH
__all__ = ['RabbitMQTLSListenerTestCase']
ST2_CI = os.environ.get('ST2_CI', 'false').lower() == 'true'
NON_SSL_LISTENER_PORT = 5672
SSL_LISTENER_PORT = 5671

@unittest2.skipIf(not ST2_CI, 'Skipping tests because ST2_CI environment variable is not set to "true"')
class RabbitMQTLSListenerTestCase(unittest2.TestCase):

    def setUp(self):
        if False:
            return 10
        cfg.CONF.set_override(name='ssl', override=False, group='messaging')
        cfg.CONF.set_override(name='ssl_keyfile', override=None, group='messaging')
        cfg.CONF.set_override(name='ssl_certfile', override=None, group='messaging')
        cfg.CONF.set_override(name='ssl_ca_certs', override=None, group='messaging')
        cfg.CONF.set_override(name='ssl_cert_reqs', override=None, group='messaging')

    def test_non_ssl_connection_on_ssl_listener_port_failure(self):
        if False:
            for i in range(10):
                print('nop')
        connection = transport_utils.get_connection(urls='amqp://guest:guest@127.0.0.1:5671/')
        expected_msg_1 = '[Errno 104]'
        expected_msg_2 = 'Socket closed'
        expected_msg_3 = 'Server unexpectedly closed connection'
        try:
            connection.connect()
        except Exception as e:
            self.assertFalse(connection.connected)
            self.assertIsInstance(e, (IOError, socket.error))
            self.assertTrue(expected_msg_1 in six.text_type(e) or expected_msg_2 in six.text_type(e) or expected_msg_3 in six.text_type(e))
        else:
            self.fail('Exception was not thrown')
            if connection:
                connection.release()

    def test_ssl_connection_on_ssl_listener_success(self):
        if False:
            for i in range(10):
                print('nop')
        ca_cert_path = os.path.join(CERTS_FIXTURES_PATH, 'ca/ca_certificate_bundle.pem')
        cfg.CONF.set_override(name='ssl', override=True, group='messaging')
        cfg.CONF.set_override(name='ssl_ca_certs', override=ca_cert_path, group='messaging')
        urls = 'amqp://guest:guest@127.0.0.1:5671/'
        connection = transport_utils.get_connection(urls=urls)
        try:
            self.assertTrue(connection.connect())
            self.assertTrue(connection.connected)
        finally:
            if connection:
                connection.release()

    def test_ssl_connection_ca_certs_provided(self):
        if False:
            while True:
                i = 10
        ca_cert_path = os.path.join(CERTS_FIXTURES_PATH, 'ca/ca_certificate_bundle.pem')
        cfg.CONF.set_override(name='ssl', override=True, group='messaging')
        cfg.CONF.set_override(name='ssl_ca_certs', override=ca_cert_path, group='messaging')
        cfg.CONF.set_override(name='ssl_cert_reqs', override='required', group='messaging')
        connection = transport_utils.get_connection(urls='amqp://guest:guest@127.0.0.1:5671/')
        try:
            self.assertTrue(connection.connect())
            self.assertTrue(connection.connected)
        finally:
            if connection:
                connection.release()
        ca_cert_path = os.path.join('/etc/ssl/certs/SecureTrust_CA.pem')
        cfg.CONF.set_override(name='ssl_cert_reqs', override='required', group='messaging')
        cfg.CONF.set_override(name='ssl_ca_certs', override=ca_cert_path, group='messaging')
        connection = transport_utils.get_connection(urls='amqp://guest:guest@127.0.0.1:5671/')
        expected_msg = '\\[SSL: CERTIFICATE_VERIFY_FAILED\\] certificate verify failed'
        self.assertRaisesRegexp(ssl.SSLError, expected_msg, connection.connect)
        ca_cert_path = os.path.join('/etc/ssl/certs/SecureTrust_CA.pem')
        cfg.CONF.set_override(name='ssl_cert_reqs', override='optional', group='messaging')
        cfg.CONF.set_override(name='ssl_ca_certs', override=ca_cert_path, group='messaging')
        connection = transport_utils.get_connection(urls='amqp://guest:guest@127.0.0.1:5671/')
        expected_msg = '\\[SSL: CERTIFICATE_VERIFY_FAILED\\] certificate verify failed'
        self.assertRaisesRegexp(ssl.SSLError, expected_msg, connection.connect)
        ca_cert_path = os.path.join('/etc/ssl/certs/SecureTrust_CA.pem')
        cfg.CONF.set_override(name='ssl_cert_reqs', override='none', group='messaging')
        cfg.CONF.set_override(name='ssl_ca_certs', override=ca_cert_path, group='messaging')
        connection = transport_utils.get_connection(urls='amqp://guest:guest@127.0.0.1:5671/')
        try:
            self.assertTrue(connection.connect())
            self.assertTrue(connection.connected)
        finally:
            if connection:
                connection.release()

    def test_ssl_connect_client_side_cert_authentication(self):
        if False:
            print('Hello World!')
        ssl_keyfile = os.path.join(CERTS_FIXTURES_PATH, 'client/private_key.pem')
        ssl_certfile = os.path.join(CERTS_FIXTURES_PATH, 'client/client_certificate.pem')
        ca_cert_path = os.path.join(CERTS_FIXTURES_PATH, 'ca/ca_certificate_bundle.pem')
        cfg.CONF.set_override(name='ssl_keyfile', override=ssl_keyfile, group='messaging')
        cfg.CONF.set_override(name='ssl_certfile', override=ssl_certfile, group='messaging')
        cfg.CONF.set_override(name='ssl_cert_reqs', override='required', group='messaging')
        cfg.CONF.set_override(name='ssl_ca_certs', override=ca_cert_path, group='messaging')
        connection = transport_utils.get_connection(urls='amqp://guest:guest@127.0.0.1:5671/')
        try:
            self.assertTrue(connection.connect())
            self.assertTrue(connection.connected)
        finally:
            if connection:
                connection.release()
        ssl_keyfile = os.path.join(CERTS_FIXTURES_PATH, 'client/private_key.pem')
        ssl_certfile = os.path.join(CERTS_FIXTURES_PATH, 'server/server_certificate.pem')
        ca_cert_path = os.path.join(CERTS_FIXTURES_PATH, 'ca/ca_certificate_bundle.pem')
        cfg.CONF.set_override(name='ssl_keyfile', override=ssl_keyfile, group='messaging')
        cfg.CONF.set_override(name='ssl_certfile', override=ssl_certfile, group='messaging')
        cfg.CONF.set_override(name='ssl_cert_reqs', override='required', group='messaging')
        cfg.CONF.set_override(name='ssl_ca_certs', override=ca_cert_path, group='messaging')
        connection = transport_utils.get_connection(urls='amqp://guest:guest@127.0.0.1:5671/')
        expected_msg = '\\[X509: KEY_VALUES_MISMATCH\\] key values mismatch'
        self.assertRaisesRegexp(ssl.SSLError, expected_msg, connection.connect)