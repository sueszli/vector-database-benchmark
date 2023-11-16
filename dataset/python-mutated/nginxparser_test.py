"""Test for certbot_nginx._internal.nginxparser."""
import copy
import operator
import sys
import tempfile
import unittest
from pyparsing import ParseException
import pytest
from certbot_nginx._internal.nginxparser import dump
from certbot_nginx._internal.nginxparser import dumps
from certbot_nginx._internal.nginxparser import load
from certbot_nginx._internal.nginxparser import loads
from certbot_nginx._internal.nginxparser import RawNginxParser
from certbot_nginx._internal.nginxparser import UnspacedList
from certbot_nginx._internal.tests import test_util as util
FIRST = operator.itemgetter(0)

class TestRawNginxParser(unittest.TestCase):
    """Test the raw low-level Nginx config parser."""

    def test_assignments(self):
        if False:
            print('Hello World!')
        parsed = RawNginxParser.assignment.parseString('root /test;').asList()
        assert parsed == ['root', ' ', '/test']
        parsed = RawNginxParser.assignment.parseString('root /test;foo bar;').asList()
        assert parsed == ['root', ' ', '/test'], ['foo', ' ', 'bar']

    def test_blocks(self):
        if False:
            for i in range(10):
                print('nop')
        parsed = RawNginxParser.block.parseString('foo {}').asList()
        assert parsed == [['foo', ' '], []]
        parsed = RawNginxParser.block.parseString('location /foo{}').asList()
        assert parsed == [['location', ' ', '/foo'], []]
        parsed = RawNginxParser.block.parseString('foo { bar foo ; }').asList()
        assert parsed == [['foo', ' '], [[' ', 'bar', ' ', 'foo', ' '], ' ']]

    def test_nested_blocks(self):
        if False:
            i = 10
            return i + 15
        parsed = RawNginxParser.block.parseString('foo { bar {} }').asList()
        (block, content) = parsed
        assert FIRST(content) == [[' ', 'bar', ' '], []]
        assert FIRST(block) == 'foo'

    def test_dump_as_string(self):
        if False:
            return 10
        dumped = dumps(UnspacedList([['user', ' ', 'www-data'], [['\n', 'server', ' '], [['\n    ', 'listen', ' ', '80'], ['\n    ', 'server_name', ' ', 'foo.com'], ['\n    ', 'root', ' ', '/home/ubuntu/sites/foo/'], [['\n\n    ', 'location', ' ', '/status', ' '], [['\n        ', 'check_status', ''], [['\n\n        ', 'types', ' '], [['\n            ', 'image/jpeg', ' ', 'jpg']]]]]]]]))
        assert dumped.split('\n') == 'user www-data;\nserver {\n    listen 80;\n    server_name foo.com;\n    root /home/ubuntu/sites/foo/;\n\n    location /status {\n        check_status;\n\n        types {\n            image/jpeg jpg;}}}'.split('\n')

    def test_parse_from_file(self):
        if False:
            i = 10
            return i + 15
        with util.get_data_filename('foo.conf') as path:
            with open(path) as handle:
                parsed = util.filter_comments(load(handle))
        assert parsed == [['user', 'www-data'], [['http'], [[['server'], [['listen', '*:80', 'default_server', 'ssl'], ['server_name', '*.www.foo.com', '*.www.example.com'], ['root', '/home/ubuntu/sites/foo/'], [['location', '/status'], [[['types'], [['image/jpeg', 'jpg']]]]], [['location', '~', 'case_sensitive\\.php$'], [['index', 'index.php'], ['root', '/var/root']]], [['location', '~*', 'case_insensitive\\.php$'], []], [['location', '=', 'exact_match\\.php$'], []], [['location', '^~', 'ignore_regex\\.php$'], []]]]]]]

    def test_parse_from_file2(self):
        if False:
            while True:
                i = 10
        with util.get_data_filename('edge_cases.conf') as path:
            with open(path) as handle:
                parsed = util.filter_comments(load(handle))
        assert parsed == [[['server'], [['server_name', 'simple']]], [['server'], [['server_name', 'with.if'], [['location', '~', '^/services/.+$'], [[['if', '($request_filename', '~*', '\\.(ttf|woff)$)'], [['add_header', 'Access-Control-Allow-Origin', '"*"']]]]]]], [['server'], [['server_name', 'with.complicated.headers'], [['location', '~*', '\\.(?:gif|jpe?g|png)$'], [['add_header', 'Pragma', 'public'], ['add_header', 'Cache-Control', "'public, must-revalidate, proxy-revalidate'", '"test,;{}"', 'foo'], ['blah', '"hello;world"'], ['try_files', '$uri', '@rewrites']]]]]]

    def test_parse_from_file3(self):
        if False:
            return 10
        with util.get_data_filename('multiline_quotes.conf') as path:
            with open(path) as handle:
                parsed = util.filter_comments(load(handle))
        assert parsed == [[['http'], [[['server'], [['listen', '*:443'], [['location', '/'], [['body_filter_by_lua', '\'ngx.ctx.buffered = (ngx.ctx.buffered or "") .. string.sub(ngx.arg[1], 1, 1000)\n                            if ngx.arg[2] then\n                              ngx.var.resp_body = ngx.ctx.buffered\n                            end\'']]]]]]]]

    def test_abort_on_parse_failure(self):
        if False:
            for i in range(10):
                print('nop')
        with util.get_data_filename('broken.conf') as path:
            with open(path) as handle:
                with pytest.raises(ParseException):
                    load(handle)

    def test_dump_as_file(self):
        if False:
            print('Hello World!')
        with util.get_data_filename('nginx.conf') as path:
            with open(path) as handle:
                parsed = load(handle)
        parsed[-1][-1].append(UnspacedList([['server'], [['listen', ' ', '443', ' ', 'ssl'], ['server_name', ' ', 'localhost'], ['ssl_certificate', ' ', 'cert.pem'], ['ssl_certificate_key', ' ', 'cert.key'], ['ssl_session_cache', ' ', 'shared:SSL:1m'], ['ssl_session_timeout', ' ', '5m'], ['ssl_ciphers', ' ', 'HIGH:!aNULL:!MD5'], [['location', ' ', '/'], [['root', ' ', 'html'], ['index', ' ', 'index.html', ' ', 'index.htm']]]]]))
        with tempfile.TemporaryFile(mode='w+t') as f:
            dump(parsed, f)
            f.seek(0)
            parsed_new = load(f)
        assert parsed == parsed_new

    def test_comments(self):
        if False:
            while True:
                i = 10
        with util.get_data_filename('minimalistic_comments.conf') as path:
            with open(path) as handle:
                parsed = load(handle)
        with tempfile.TemporaryFile(mode='w+t') as f:
            dump(parsed, f)
            f.seek(0)
            parsed_new = load(f)
        assert parsed == parsed_new
        assert parsed_new == [['#', " Use bar.conf when it's a full moon!"], ['include', 'foo.conf'], ['#', ' Kilroy was here'], ['check_status'], [['server'], [['#', ''], ['#', " Don't forget to open up your firewall!"], ['#', ''], ['listen', '1234'], ['#', ' listen 80;']]]]

    def test_issue_518(self):
        if False:
            for i in range(10):
                print('nop')
        parsed = loads('if ($http_accept ~* "webp") { set $webp "true"; }')
        assert parsed == [[['if', '($http_accept', '~*', '"webp")'], [['set', '$webp', '"true"']]]]

    def test_comment_in_block(self):
        if False:
            while True:
                i = 10
        parsed = loads('http {\n          # server{\n          }')
        assert parsed == [[['http'], [['#', ' server{']]]]

    def test_access_log(self):
        if False:
            return 10
        parsed = loads('access_log syslog:server=unix:/dev/log,facility=auth,tag=nginx_post,severity=info custom;')
        assert parsed == [['access_log', 'syslog:server=unix:/dev/log,facility=auth,tag=nginx_post,severity=info', 'custom']]

    def test_add_header(self):
        if False:
            while True:
                i = 10
        parsed = loads('add_header Cache-Control no-cache,no-store,must-revalidate,max-age=0;')
        assert parsed == [['add_header', 'Cache-Control', 'no-cache,no-store,must-revalidate,max-age=0']]

    def test_map_then_assignment_in_block(self):
        if False:
            return 10
        test_str = 'http {\n            map $http_upgrade $connection_upgrade {\n              default upgrade;\n              \'\'      close;\n              "~Opera Mini" 1;\n              *.example.com 1;\n            }\n            one;\n        }'
        parsed = loads(test_str)
        assert parsed == [[['http'], [[['map', '$http_upgrade', '$connection_upgrade'], [['default', 'upgrade'], ["''", 'close'], ['"~Opera Mini"', '1'], ['*.example.com', '1']]], ['one']]]]

    def test_variable_name(self):
        if False:
            return 10
        parsed = loads('try_files /typo3temp/tx_ncstaticfilecache/$host${request_uri}index.html @nocache;')
        assert parsed == [['try_files', '/typo3temp/tx_ncstaticfilecache/$host${request_uri}index.html', '@nocache']]

    def test_weird_blocks(self):
        if False:
            return 10
        test = '\n            if ($http_user_agent ~ MSIE) {\n                rewrite ^(.*)$ /msie/$1 break;\n            }\n\n            if ($http_cookie ~* "id=([^;]+)(?:;|$)") {\n               set $id $1;\n            }\n\n            if ($request_method = POST) {\n               return 405;\n            }\n\n            if ($request_method) {\n               return 403;\n            }\n\n            if ($args ~ post=140){\n              rewrite ^ http://example.com/;\n            }\n\n            location ~ ^/users/(.+\\.(?:gif|jpe?g|png))$ {\n              alias /data/w3/images/$1;\n            }\n\n            proxy_set_header X-Origin-URI ${scheme}://${http_host}/$request_uri;\n        '
        parsed = loads(test)
        assert parsed == [[['if', '($http_user_agent', '~', 'MSIE)'], [['rewrite', '^(.*)$', '/msie/$1', 'break']]], [['if', '($http_cookie', '~*', '"id=([^;]+)(?:;|$)")'], [['set', '$id', '$1']]], [['if', '($request_method', '=', 'POST)'], [['return', '405']]], [['if', '($request_method)'], [['return', '403']]], [['if', '($args', '~', 'post=140)'], [['rewrite', '^', 'http://example.com/']]], [['location', '~', '^/users/(.+\\.(?:gif|jpe?g|png))$'], [['alias', '/data/w3/images/$1']]], ['proxy_set_header', 'X-Origin-URI', '${scheme}://${http_host}/$request_uri']]

    def test_edge_cases(self):
        if False:
            i = 10
            return i + 15
        parsed = loads('"hello\\""; # blah "heh heh"')
        assert parsed == [['"hello\\""'], ['#', ' blah "heh heh"']]
        parsed = loads('if ($http_cookie ~* "id=([^;]+)(?:;|$)") { # blah )\n            }')
        assert parsed == [[['if', '($http_cookie', '~*', '"id=([^;]+)(?:;|$)")'], [['#', ' blah )']]]]
        test = '\n            one"test";\n            ("two");\n            "test")red;\n            "test")"blue";\n            "test")"three;\n            (one"test")one;\n            one";\n            one"test;\n            one"test"one;\n        '
        parsed = loads(test)
        assert parsed == [['one"test"'], ['("two")'], ['"test")red'], ['"test")"blue"'], ['"test")"three'], ['(one"test")one'], ['one"'], ['one"test'], ['one"test"one']]
        with pytest.raises(ParseException):
            loads('"test"one;')
        with pytest.raises(ParseException):
            loads('"test;')
        test = '\n            server_name foo.example.com bar.example.com                         baz.example.com qux.example.com;\n            server_name foo.example.com bar.example.com\n                        baz.example.com qux.example.com;\n        '
        parsed = loads(test)
        assert parsed == [['server_name', 'foo.example.com', 'bar.example.com', 'baz.example.com', 'qux.example.com'], ['server_name', 'foo.example.com', 'bar.example.com', 'baz.example.com', 'qux.example.com']]
        parsed = loads('directive $var ${var} $ ${};')
        assert parsed == [['directive', '$var', '${var}', '$', '${}']]
        with pytest.raises(ParseException):
            loads('server {server_name test.com};')
        assert loads('blag${dfgdfg};') == [['blag${dfgdfg}']]
        with pytest.raises(ParseException):
            loads('blag${dfgdf{g};')
        parsed = loads('')
        assert parsed == []

class TestUnspacedList(unittest.TestCase):
    """Test the UnspacedList data structure"""

    def setUp(self):
        if False:
            while True:
                i = 10
        self.a = ['\n    ', 'things', ' ', 'quirk']
        self.b = ['y', ' ']
        self.l = self.a[:]
        self.l2 = self.b[:]
        self.ul = UnspacedList(self.l)
        self.ul2 = UnspacedList(self.l2)

    def test_construction(self):
        if False:
            print('Hello World!')
        assert self.ul == ['things', 'quirk']
        assert self.ul2 == ['y']

    def test_append(self):
        if False:
            i = 10
            return i + 15
        ul3 = copy.deepcopy(self.ul)
        ul3.append('wise')
        assert ul3 == ['things', 'quirk', 'wise']
        assert ul3.spaced == self.a + ['wise']

    def test_add(self):
        if False:
            for i in range(10):
                print('nop')
        ul3 = self.ul + self.ul2
        assert ul3 == ['things', 'quirk', 'y']
        assert ul3.spaced == self.a + self.b
        assert self.ul.spaced == self.a
        ul3 = self.ul + self.l2
        assert ul3 == ['things', 'quirk', 'y']
        assert ul3.spaced == self.a + self.b

    def test_extend(self):
        if False:
            while True:
                i = 10
        ul3 = copy.deepcopy(self.ul)
        ul3.extend(self.ul2)
        assert ul3 == ['things', 'quirk', 'y']
        assert ul3.spaced == self.a + self.b
        assert self.ul.spaced == self.a

    def test_set(self):
        if False:
            for i in range(10):
                print('nop')
        ul3 = copy.deepcopy(self.ul)
        ul3[0] = 'zither'
        l = ['\n ', 'zather', 'zest']
        ul3[1] = UnspacedList(l)
        assert ul3 == ['zither', ['zather', 'zest']]
        assert ul3.spaced == [self.a[0], 'zither', ' ', l]

    def test_get(self):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(IndexError):
            self.ul2.__getitem__(2)
        with pytest.raises(IndexError):
            self.ul2.__getitem__(-3)

    def test_insert(self):
        if False:
            return 10
        x = UnspacedList([['\n    ', 'listen', '       ', '69.50.225.155:9000'], ['\n    ', 'listen', '       ', '127.0.0.1'], ['\n    ', 'server_name', ' ', '.example.com'], ['\n    ', 'server_name', ' ', 'example.*'], '\n', ['listen', ' ', '5001', ' ', 'ssl']])
        x.insert(5, 'FROGZ')
        assert x == [['listen', '69.50.225.155:9000'], ['listen', '127.0.0.1'], ['server_name', '.example.com'], ['server_name', 'example.*'], ['listen', '5001', 'ssl'], 'FROGZ']
        assert x.spaced == [['\n    ', 'listen', '       ', '69.50.225.155:9000'], ['\n    ', 'listen', '       ', '127.0.0.1'], ['\n    ', 'server_name', ' ', '.example.com'], ['\n    ', 'server_name', ' ', 'example.*'], '\n', ['listen', ' ', '5001', ' ', 'ssl'], 'FROGZ']

    def test_rawlists(self):
        if False:
            print('Hello World!')
        ul3 = copy.deepcopy(self.ul)
        ul3.insert(0, 'some')
        ul3.append('why')
        ul3.extend(['did', 'whether'])
        del ul3[2]
        assert ul3 == ['some', 'things', 'why', 'did', 'whether']

    def test_is_dirty(self):
        if False:
            print('Hello World!')
        assert self.ul2.is_dirty() is False
        ul3 = UnspacedList([])
        ul3.append(self.ul)
        assert self.ul.is_dirty() is False
        assert ul3.is_dirty() is True
        ul4 = UnspacedList([[1], [2, 3, 4]])
        assert ul4.is_dirty() is False
        ul4[1][2] = 5
        assert ul4.is_dirty() is True
if __name__ == '__main__':
    sys.exit(pytest.main(sys.argv[1:] + [__file__]))