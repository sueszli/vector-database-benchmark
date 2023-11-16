from typing import cast
import idna
from OpenSSL import SSL
from synapse.config._base import Config, RootConfig
from synapse.config.homeserver import HomeServerConfig
from synapse.config.tls import ConfigError, TlsConfig
from synapse.crypto.context_factory import FederationPolicyForHTTPS, SSLClientConnectionCreator
from synapse.types import JsonDict
from tests.unittest import TestCase

class FakeServer(Config):
    section = 'server'

    def has_tls_listener(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return False

class TestConfig(RootConfig):
    config_classes = [FakeServer, TlsConfig]

class TLSConfigTests(TestCase):

    def test_tls_client_minimum_default(self) -> None:
        if False:
            while True:
                i = 10
        '\n        The default client TLS version is 1.0.\n        '
        config: JsonDict = {}
        t = TestConfig()
        t.tls.read_config(config, config_dir_path='', data_dir_path='')
        self.assertEqual(t.tls.federation_client_minimum_tls_version, '1')

    def test_tls_client_minimum_set(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        The default client TLS version can be set to 1.0, 1.1, and 1.2.\n        '
        config: JsonDict = {'federation_client_minimum_tls_version': 1}
        t = TestConfig()
        t.tls.read_config(config, config_dir_path='', data_dir_path='')
        self.assertEqual(t.tls.federation_client_minimum_tls_version, '1')
        config = {'federation_client_minimum_tls_version': 1.1}
        t = TestConfig()
        t.tls.read_config(config, config_dir_path='', data_dir_path='')
        self.assertEqual(t.tls.federation_client_minimum_tls_version, '1.1')
        config = {'federation_client_minimum_tls_version': 1.2}
        t = TestConfig()
        t.tls.read_config(config, config_dir_path='', data_dir_path='')
        self.assertEqual(t.tls.federation_client_minimum_tls_version, '1.2')
        config = {'federation_client_minimum_tls_version': '1'}
        t = TestConfig()
        t.tls.read_config(config, config_dir_path='', data_dir_path='')
        self.assertEqual(t.tls.federation_client_minimum_tls_version, '1')
        config = {'federation_client_minimum_tls_version': '1.2'}
        t = TestConfig()
        t.tls.read_config(config, config_dir_path='', data_dir_path='')
        self.assertEqual(t.tls.federation_client_minimum_tls_version, '1.2')

    def test_tls_client_minimum_1_point_3_missing(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        If TLS 1.3 support is missing and it's configured, it will raise a\n        ConfigError.\n        "
        if hasattr(SSL, 'OP_NO_TLSv1_3'):
            OP_NO_TLSv1_3 = SSL.OP_NO_TLSv1_3
            delattr(SSL, 'OP_NO_TLSv1_3')
            self.addCleanup(setattr, SSL, 'SSL.OP_NO_TLSv1_3', OP_NO_TLSv1_3)
            assert not hasattr(SSL, 'OP_NO_TLSv1_3')
        config: JsonDict = {'federation_client_minimum_tls_version': 1.3}
        t = TestConfig()
        with self.assertRaises(ConfigError) as e:
            t.tls.read_config(config, config_dir_path='', data_dir_path='')
        self.assertEqual(e.exception.args[0], 'federation_client_minimum_tls_version cannot be 1.3, your OpenSSL does not support it')

    def test_tls_client_minimum_1_point_3_exists(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        If TLS 1.3 support exists and it's configured, it will be settable.\n        "
        if not hasattr(SSL, 'OP_NO_TLSv1_3'):
            SSL.OP_NO_TLSv1_3 = 0
            self.addCleanup(lambda : delattr(SSL, 'OP_NO_TLSv1_3'))
            assert hasattr(SSL, 'OP_NO_TLSv1_3')
        config: JsonDict = {'federation_client_minimum_tls_version': 1.3}
        t = TestConfig()
        t.tls.read_config(config, config_dir_path='', data_dir_path='')
        self.assertEqual(t.tls.federation_client_minimum_tls_version, '1.3')

    def test_tls_client_minimum_set_passed_through_1_2(self) -> None:
        if False:
            print('Hello World!')
        '\n        The configured TLS version is correctly configured by the ContextFactory.\n        '
        config: JsonDict = {'federation_client_minimum_tls_version': 1.2}
        t = TestConfig()
        t.tls.read_config(config, config_dir_path='', data_dir_path='')
        cf = FederationPolicyForHTTPS(cast(HomeServerConfig, t))
        options = _get_ssl_context_options(cf._verify_ssl_context)
        self.assertNotEqual(options & SSL.OP_NO_TLSv1, 0)
        self.assertNotEqual(options & SSL.OP_NO_TLSv1_1, 0)
        self.assertEqual(options & SSL.OP_NO_TLSv1_2, 0)

    def test_tls_client_minimum_set_passed_through_1_0(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        The configured TLS version is correctly configured by the ContextFactory.\n        '
        config: JsonDict = {'federation_client_minimum_tls_version': 1}
        t = TestConfig()
        t.tls.read_config(config, config_dir_path='', data_dir_path='')
        cf = FederationPolicyForHTTPS(cast(HomeServerConfig, t))
        options = _get_ssl_context_options(cf._verify_ssl_context)
        self.assertEqual(options & SSL.OP_NO_TLSv1, 0)
        self.assertEqual(options & SSL.OP_NO_TLSv1_1, 0)
        self.assertEqual(options & SSL.OP_NO_TLSv1_2, 0)

    def test_whitelist_idna_failure(self) -> None:
        if False:
            print('Hello World!')
        '\n        The federation certificate whitelist will not allow IDNA domain names.\n        '
        config: JsonDict = {'federation_certificate_verification_whitelist': ['example.com', '*.ドメイン.テスト']}
        t = TestConfig()
        e = self.assertRaises(ConfigError, t.tls.read_config, config, config_dir_path='', data_dir_path='')
        self.assertIn('IDNA domain names', str(e))

    def test_whitelist_idna_result(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        The federation certificate whitelist will match on IDNA encoded names.\n        '
        config: JsonDict = {'federation_certificate_verification_whitelist': ['example.com', '*.xn--eckwd4c7c.xn--zckzah']}
        t = TestConfig()
        t.tls.read_config(config, config_dir_path='', data_dir_path='')
        cf = FederationPolicyForHTTPS(cast(HomeServerConfig, t))
        opts = cf.get_options(b'notexample.com')
        assert isinstance(opts, SSLClientConnectionCreator)
        self.assertTrue(opts._verifier._verify_certs)
        opts = cf.get_options(idna.encode('テスト.ドメイン.テスト'))
        assert isinstance(opts, SSLClientConnectionCreator)
        self.assertFalse(opts._verifier._verify_certs)

def _get_ssl_context_options(ssl_context: SSL.Context) -> int:
    if False:
        print('Hello World!')
    'get the options bits from an openssl context object'
    return SSL._lib.SSL_CTX_get_options(ssl_context._context)