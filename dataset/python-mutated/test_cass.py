import ssl
from unittest import TestCase
from redash.query_runner.cass import generate_ssl_options_dict

class TestCassandra(TestCase):

    def test_generate_ssl_options_dict_creates_plain_protocol_dict(self):
        if False:
            i = 10
            return i + 15
        expected = {'ssl_version': ssl.PROTOCOL_TLSv1_2}
        actual = generate_ssl_options_dict('PROTOCOL_TLSv1_2')
        self.assertDictEqual(expected, actual)

    def test_generate_ssl_options_dict_creates_certificate_dict(self):
        if False:
            for i in range(10):
                print('nop')
        expected = {'ssl_version': ssl.PROTOCOL_TLSv1_2, 'ca_certs': 'some/path', 'cert_reqs': ssl.CERT_REQUIRED}
        actual = generate_ssl_options_dict('PROTOCOL_TLSv1_2', 'some/path')
        self.assertDictEqual(expected, actual)