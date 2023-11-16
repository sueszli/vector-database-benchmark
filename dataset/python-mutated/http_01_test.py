"""Tests for certbot_nginx._internal.http_01"""
import sys
import unittest
from unittest import mock
import josepy as jose
import pytest
from acme import challenges
from acme import messages
from certbot import achallenges
from certbot.tests import acme_util
from certbot.tests import util as test_util
from certbot_nginx._internal.obj import Addr
from certbot_nginx._internal.tests import test_util as util
AUTH_KEY = jose.JWKRSA.load(test_util.load_vector('rsa512_key.pem'))

class HttpPerformTest(util.NginxTest):
    """Test the NginxHttp01 challenge."""
    account_key = AUTH_KEY
    achalls = [achallenges.KeyAuthorizationAnnotatedChallenge(challb=acme_util.chall_to_challb(challenges.HTTP01(token=b'kNdwjwOeX0I_A8DXt9Msmg'), messages.STATUS_PENDING), domain='www.example.com', account_key=account_key), achallenges.KeyAuthorizationAnnotatedChallenge(challb=acme_util.chall_to_challb(challenges.HTTP01(token=b'\xba\xa9\xda?<m\xaewmx\xea\xad\xadv\xf4\x02\xc9y\x80\xe2_X\t\xe7\xc7\xa4\t\xca\xf7&\x945'), messages.STATUS_PENDING), domain='ipv6.com', account_key=account_key), achallenges.KeyAuthorizationAnnotatedChallenge(challb=acme_util.chall_to_challb(challenges.HTTP01(token=b'\x8c\x8a\xbf_-f\\cw\xee\xd6\xf8/\xa5\xe3\xfd\xeb9\xf1\xf5\xb9\xefVM\xc9w\xa4u\x9c\xe1\x87\xb4'), messages.STATUS_PENDING), domain='www.example.org', account_key=account_key), achallenges.KeyAuthorizationAnnotatedChallenge(challb=acme_util.chall_to_challb(challenges.HTTP01(token=b'kNdwjxOeX0I_A8DXt9Msmg'), messages.STATUS_PENDING), domain='migration.com', account_key=account_key), achallenges.KeyAuthorizationAnnotatedChallenge(challb=acme_util.chall_to_challb(challenges.HTTP01(token=b'kNdwjxOeX0I_A8DXt9Msmg'), messages.STATUS_PENDING), domain='ipv6ssl.com', account_key=account_key)]

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        config = self.get_nginx_configurator(self.config_path, self.config_dir, self.work_dir, self.logs_dir)
        from certbot_nginx._internal import http_01
        self.http01 = http_01.NginxHttp01(config)

    def test_perform0(self):
        if False:
            print('Hello World!')
        responses = self.http01.perform()
        assert [] == responses

    @mock.patch('certbot_nginx._internal.configurator.NginxConfigurator.save')
    def test_perform1(self, mock_save):
        if False:
            for i in range(10):
                print('nop')
        self.http01.add_chall(self.achalls[0])
        response = self.achalls[0].response(self.account_key)
        responses = self.http01.perform()
        assert [response] == responses
        assert mock_save.call_count == 1

    def test_perform2(self):
        if False:
            while True:
                i = 10
        acme_responses = []
        for achall in self.achalls:
            self.http01.add_chall(achall)
            acme_responses.append(achall.response(self.account_key))
        http_responses = self.http01.perform()
        assert len(http_responses) == 5
        for i in range(5):
            assert http_responses[i] == acme_responses[i]

    def test_mod_config(self):
        if False:
            print('Hello World!')
        self.http01.add_chall(self.achalls[0])
        self.http01.add_chall(self.achalls[2])
        self.http01._mod_config()
        self.http01.configurator.save()
        self.http01.configurator.parser.load()

    @mock.patch('certbot_nginx._internal.parser.NginxParser.add_server_directives')
    def test_mod_config_http_and_https(self, mock_add_server_directives):
        if False:
            for i in range(10):
                print('nop')
        'A server_name with both HTTP and HTTPS vhosts should get modded in both vhosts'
        self.configuration.https_port = 443
        self.http01.add_chall(self.achalls[3])
        self.http01._mod_config()
        assert mock_add_server_directives.call_count == 4

    @mock.patch('certbot_nginx._internal.parser.nginxparser.dump')
    @mock.patch('certbot_nginx._internal.parser.NginxParser.add_server_directives')
    def test_mod_config_only_https(self, mock_add_server_directives, mock_dump):
        if False:
            i = 10
            return i + 15
        'A server_name with only an HTTPS vhost should get modded'
        self.http01.add_chall(self.achalls[4])
        self.http01._mod_config()
        assert mock_add_server_directives.call_count == 2
        assert mock_dump.call_args[0][0] != []

    @mock.patch('certbot_nginx._internal.parser.NginxParser.add_server_directives')
    def test_mod_config_deduplicate(self, mock_add_server_directives):
        if False:
            return 10
        'A vhost that appears in both HTTP and HTTPS vhosts only gets modded once'
        achall = achallenges.KeyAuthorizationAnnotatedChallenge(challb=acme_util.chall_to_challb(challenges.HTTP01(token=b'kNdwjxOeX0I_A8DXt9Msmg'), messages.STATUS_PENDING), domain='ssl.both.com', account_key=AUTH_KEY)
        self.http01.add_chall(achall)
        self.http01._mod_config()
        assert mock_add_server_directives.call_count == 5 * 2

    def test_mod_config_insert_bucket_directive(self):
        if False:
            i = 10
            return i + 15
        nginx_conf = self.http01.configurator.parser.abs_path('nginx.conf')
        expected = ['server_names_hash_bucket_size', '128']
        original_conf = self.http01.configurator.parser.parsed[nginx_conf]
        assert not util.contains_at_depth(original_conf, expected, 2)
        self.http01.add_chall(self.achalls[0])
        self.http01._mod_config()
        self.http01.configurator.save()
        self.http01.configurator.parser.load()
        generated_conf = self.http01.configurator.parser.parsed[nginx_conf]
        assert util.contains_at_depth(generated_conf, expected, 2)

    def test_mod_config_update_bucket_directive_in_included_file(self):
        if False:
            while True:
                i = 10
        example_com_loc = self.http01.configurator.parser.abs_path('sites-enabled/example.com')
        with open(example_com_loc) as f:
            original_example_com = f.read()
        modified_example_com = 'server_names_hash_bucket_size 64;\n' + original_example_com
        with open(example_com_loc, 'w') as f:
            f.write(modified_example_com)
        self.http01.configurator.parser.load()
        self.http01.add_chall(self.achalls[0])
        self.http01._mod_config()
        self.http01.configurator.save()
        self.http01.configurator.parser.load()
        expected = ['server_names_hash_bucket_size', '128']
        nginx_conf_loc = self.http01.configurator.parser.abs_path('nginx.conf')
        nginx_conf = self.http01.configurator.parser.parsed[nginx_conf_loc]
        assert not util.contains_at_depth(nginx_conf, expected, 2)
        generated_conf = self.http01.configurator.parser.parsed[example_com_loc]
        assert util.contains_at_depth(generated_conf, expected, 0)
        with open(example_com_loc, 'w') as f:
            f.write(original_example_com)
        self.http01.configurator.parser.load()

    @mock.patch('certbot_nginx._internal.configurator.NginxConfigurator.ipv6_info')
    def test_default_listen_addresses_no_memoization(self, ipv6_info):
        if False:
            while True:
                i = 10
        ipv6_info.return_value = (True, True)
        self.http01._default_listen_addresses()
        assert ipv6_info.call_count == 1
        ipv6_info.return_value = (False, False)
        self.http01._default_listen_addresses()
        assert ipv6_info.call_count == 2

    @mock.patch('certbot_nginx._internal.configurator.NginxConfigurator.ipv6_info')
    def test_default_listen_addresses_t_t(self, ipv6_info):
        if False:
            return 10
        ipv6_info.return_value = (True, True)
        addrs = self.http01._default_listen_addresses()
        http_addr = Addr.fromstring('80')
        http_ipv6_addr = Addr.fromstring('[::]:80')
        assert addrs == [http_addr, http_ipv6_addr]

    @mock.patch('certbot_nginx._internal.configurator.NginxConfigurator.ipv6_info')
    def test_default_listen_addresses_t_f(self, ipv6_info):
        if False:
            for i in range(10):
                print('nop')
        ipv6_info.return_value = (True, False)
        addrs = self.http01._default_listen_addresses()
        http_addr = Addr.fromstring('80')
        http_ipv6_addr = Addr.fromstring('[::]:80 ipv6only=on')
        assert addrs == [http_addr, http_ipv6_addr]

    @mock.patch('certbot_nginx._internal.configurator.NginxConfigurator.ipv6_info')
    def test_default_listen_addresses_f_f(self, ipv6_info):
        if False:
            i = 10
            return i + 15
        ipv6_info.return_value = (False, False)
        addrs = self.http01._default_listen_addresses()
        http_addr = Addr.fromstring('80')
        assert addrs == [http_addr]
if __name__ == '__main__':
    sys.exit(pytest.main(sys.argv[1:] + [__file__]))