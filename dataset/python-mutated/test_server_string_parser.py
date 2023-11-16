from sslyze.cli.server_string_parser import CommandLineServerStringParser

class TestCommandLineServerStringParser:

    def test(self):
        if False:
            return 10
        server_string = 'www.google.com'
        (hostname, ip_address, port) = CommandLineServerStringParser.parse_server_string(server_string)
        assert 'www.google.com' == hostname
        assert not port
        assert not ip_address

    def test_with_port(self):
        if False:
            return 10
        server_string = 'www.google.com:443'
        (hostname, ip_address, port) = CommandLineServerStringParser.parse_server_string(server_string)
        assert 'www.google.com' == hostname
        assert 443 == port
        assert not ip_address

    def test_ipv4_as_hint(self):
        if False:
            print('Hello World!')
        server_string = 'www.google.com{192.168.2.1}'
        (hostname, ip_address, port) = CommandLineServerStringParser.parse_server_string(server_string)
        assert 'www.google.com' == hostname
        assert not port
        assert '192.168.2.1' == ip_address

    def test_ipv4_as_hint_with_port(self):
        if False:
            for i in range(10):
                print('nop')
        server_string = 'www.google.com:443{192.168.2.1}'
        (hostname, ip_address, port) = CommandLineServerStringParser.parse_server_string(server_string)
        assert 'www.google.com' == hostname
        assert 443 == port
        assert '192.168.2.1' == ip_address

    def test_ipv6(self):
        if False:
            print('Hello World!')
        server_string = '[2604:5500:c370:e100:15ba:f57b:e10e:50c1]'
        (hostname, ip_address, port) = CommandLineServerStringParser.parse_server_string(server_string)
        assert '2604:5500:c370:e100:15ba:f57b:e10e:50c1' == hostname
        assert not port
        assert not ip_address

    def test_ipv6_with_port(self):
        if False:
            return 10
        server_string = '[2604:5500:c370:e100:15ba:f57b:e10e:50c1]:443'
        (hostname, ip_address, port) = CommandLineServerStringParser.parse_server_string(server_string)
        assert '2604:5500:c370:e100:15ba:f57b:e10e:50c1' == hostname
        assert 443 == port
        assert not ip_address

    def test_ipv6_as_hint(self):
        if False:
            i = 10
            return i + 15
        server_string = 'www.google.com{[2604:5500:c370:e100:15ba:f57b:e10e:50c1]}'
        (hostname, ip_address, port) = CommandLineServerStringParser.parse_server_string(server_string)
        assert 'www.google.com' == hostname
        assert not port
        assert '2604:5500:c370:e100:15ba:f57b:e10e:50c1' == ip_address

    def test_ipv6_as_hint_with_port(self):
        if False:
            while True:
                i = 10
        server_string = 'www.google.com:443{[2604:5500:c370:e100:15ba:f57b:e10e:50c1]}'
        (hostname, ip_address, port) = CommandLineServerStringParser.parse_server_string(server_string)
        assert 'www.google.com' == hostname
        assert 443 == port
        assert '2604:5500:c370:e100:15ba:f57b:e10e:50c1' == ip_address