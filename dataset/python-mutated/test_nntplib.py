import io
import socket
import datetime
import textwrap
import unittest
import functools
import contextlib
import nntplib
import os.path
import re
import threading
from test import support
from test.support import socket_helper
from nntplib import NNTP, GroupInfo
from unittest.mock import patch
try:
    import ssl
except ImportError:
    ssl = None
certfile = os.path.join(os.path.dirname(__file__), 'keycert3.pem')
if ssl is not None:
    SSLError = ssl.SSLError
else:

    class SSLError(Exception):
        """Non-existent exception class when we lack SSL support."""
        reason = 'This will never be raised.'

class NetworkedNNTPTestsMixin:
    ssl_context = None

    def test_welcome(self):
        if False:
            i = 10
            return i + 15
        welcome = self.server.getwelcome()
        self.assertEqual(str, type(welcome))

    def test_help(self):
        if False:
            while True:
                i = 10
        (resp, lines) = self.server.help()
        self.assertTrue(resp.startswith('100 '), resp)
        for line in lines:
            self.assertEqual(str, type(line))

    def test_list(self):
        if False:
            for i in range(10):
                print('nop')
        (resp, groups) = self.server.list()
        if len(groups) > 0:
            self.assertEqual(GroupInfo, type(groups[0]))
            self.assertEqual(str, type(groups[0].group))

    def test_list_active(self):
        if False:
            while True:
                i = 10
        (resp, groups) = self.server.list(self.GROUP_PAT)
        if len(groups) > 0:
            self.assertEqual(GroupInfo, type(groups[0]))
            self.assertEqual(str, type(groups[0].group))

    def test_unknown_command(self):
        if False:
            print('Hello World!')
        with self.assertRaises(nntplib.NNTPPermanentError) as cm:
            self.server._shortcmd('XYZZY')
        resp = cm.exception.response
        self.assertTrue(resp.startswith('500 '), resp)

    def test_newgroups(self):
        if False:
            return 10
        dt = datetime.date.today() - datetime.timedelta(days=7)
        (resp, groups) = self.server.newgroups(dt)
        if len(groups) > 0:
            self.assertIsInstance(groups[0], GroupInfo)
            self.assertIsInstance(groups[0].group, str)

    def test_description(self):
        if False:
            print('Hello World!')

        def _check_desc(desc):
            if False:
                return 10
            self.assertIsInstance(desc, str)
            self.assertNotIn(self.GROUP_NAME, desc)
        desc = self.server.description(self.GROUP_NAME)
        _check_desc(desc)
        self.assertIn(self.DESC, desc)
        desc = self.server.description(self.GROUP_PAT)
        _check_desc(desc)
        desc = self.server.description('zk.brrtt.baz')
        self.assertEqual(desc, '')

    def test_descriptions(self):
        if False:
            return 10
        (resp, descs) = self.server.descriptions(self.GROUP_PAT)
        self.assertTrue(resp.startswith('215 ') or resp.startswith('282 '), resp)
        self.assertIsInstance(descs, dict)
        desc = descs[self.GROUP_NAME]
        self.assertEqual(desc, self.server.description(self.GROUP_NAME))

    def test_group(self):
        if False:
            while True:
                i = 10
        result = self.server.group(self.GROUP_NAME)
        self.assertEqual(5, len(result))
        (resp, count, first, last, group) = result
        self.assertEqual(group, self.GROUP_NAME)
        self.assertIsInstance(count, int)
        self.assertIsInstance(first, int)
        self.assertIsInstance(last, int)
        self.assertLessEqual(first, last)
        self.assertTrue(resp.startswith('211 '), resp)

    def test_date(self):
        if False:
            for i in range(10):
                print('nop')
        (resp, date) = self.server.date()
        self.assertIsInstance(date, datetime.datetime)
        self.assertGreaterEqual(date.year, 1995)
        self.assertLessEqual(date.year, 2030)

    def _check_art_dict(self, art_dict):
        if False:
            while True:
                i = 10
        self.assertIsInstance(art_dict, dict)
        self.assertGreaterEqual(art_dict.keys(), {'subject', 'from', 'date', 'message-id', 'references', ':bytes', ':lines'})
        for v in art_dict.values():
            self.assertIsInstance(v, (str, type(None)))

    def test_xover(self):
        if False:
            i = 10
            return i + 15
        (resp, count, first, last, name) = self.server.group(self.GROUP_NAME)
        (resp, lines) = self.server.xover(last - 5, last)
        if len(lines) == 0:
            self.skipTest('no articles retrieved')
        (art_num, art_dict) = lines[0]
        self.assertGreaterEqual(art_num, last - 5)
        self.assertLessEqual(art_num, last)
        self._check_art_dict(art_dict)

    @unittest.skipIf(True, 'temporarily skipped until a permanent solution is found for issue #28971')
    def test_over(self):
        if False:
            print('Hello World!')
        (resp, count, first, last, name) = self.server.group(self.GROUP_NAME)
        start = last - 10
        (resp, lines) = self.server.over((start, None))
        (art_num, art_dict) = lines[0]
        self._check_art_dict(art_dict)
        (resp, lines) = self.server.over((start, last))
        (art_num, art_dict) = lines[-1]
        self.assertGreaterEqual(art_num, start)
        self.assertLessEqual(art_num, last)
        self._check_art_dict(art_dict)

    def test_xhdr(self):
        if False:
            while True:
                i = 10
        (resp, count, first, last, name) = self.server.group(self.GROUP_NAME)
        (resp, lines) = self.server.xhdr('subject', last)
        for line in lines:
            self.assertEqual(str, type(line[1]))

    def check_article_resp(self, resp, article, art_num=None):
        if False:
            i = 10
            return i + 15
        self.assertIsInstance(article, nntplib.ArticleInfo)
        if art_num is not None:
            self.assertEqual(article.number, art_num)
        for line in article.lines:
            self.assertIsInstance(line, bytes)
        self.assertNotIn(article.lines[-1], (b'.', b'.\n', b'.\r\n'))

    @unittest.skipIf(True, 'FIXME: see bpo-32128')
    def test_article_head_body(self):
        if False:
            for i in range(10):
                print('nop')
        (resp, count, first, last, name) = self.server.group(self.GROUP_NAME)
        for art_num in (last, first, last - 1):
            try:
                (resp, head) = self.server.head(art_num)
            except nntplib.NNTPTemporaryError as e:
                if not e.response.startswith('423 '):
                    raise
                continue
            break
        else:
            self.skipTest('could not find a suitable article number')
        self.assertTrue(resp.startswith('221 '), resp)
        self.check_article_resp(resp, head, art_num)
        (resp, body) = self.server.body(art_num)
        self.assertTrue(resp.startswith('222 '), resp)
        self.check_article_resp(resp, body, art_num)
        (resp, article) = self.server.article(art_num)
        self.assertTrue(resp.startswith('220 '), resp)
        self.check_article_resp(resp, article, art_num)
        denylist = lambda line: line.startswith(b'X-Antivirus')
        filtered_head_lines = [line for line in head.lines if not denylist(line)]
        filtered_lines = [line for line in article.lines if not denylist(line)]
        self.assertEqual(filtered_lines, filtered_head_lines + [b''] + body.lines)

    def test_capabilities(self):
        if False:
            print('Hello World!')

        def _check_caps(caps):
            if False:
                for i in range(10):
                    print('nop')
            caps_list = caps['LIST']
            self.assertIsInstance(caps_list, (list, tuple))
            self.assertIn('OVERVIEW.FMT', caps_list)
        self.assertGreaterEqual(self.server.nntp_version, 2)
        _check_caps(self.server.getcapabilities())
        (resp, caps) = self.server.capabilities()
        _check_caps(caps)

    def test_zlogin(self):
        if False:
            i = 10
            return i + 15
        baduser = 'notarealuser'
        badpw = 'notarealpassword'
        self.assertRaises(nntplib.NNTPError, self.server.login, user=baduser, password=badpw, usenetrc=False)

    def test_zzquit(self):
        if False:
            return 10
        cls = type(self)
        try:
            self.server.quit()
        finally:
            cls.server = None

    @classmethod
    def wrap_methods(cls):
        if False:
            print('Hello World!')

        def wrap_meth(meth):
            if False:
                print('Hello World!')

            @functools.wraps(meth)
            def wrapped(self):
                if False:
                    i = 10
                    return i + 15
                with socket_helper.transient_internet(self.NNTP_HOST):
                    meth(self)
            return wrapped
        for name in dir(cls):
            if not name.startswith('test_'):
                continue
            meth = getattr(cls, name)
            if not callable(meth):
                continue
            setattr(cls, name, wrap_meth(meth))

    def test_timeout(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            self.NNTP_CLASS(self.NNTP_HOST, timeout=0, usenetrc=False)

    def test_with_statement(self):
        if False:
            print('Hello World!')

        def is_connected():
            if False:
                i = 10
                return i + 15
            if not hasattr(server, 'file'):
                return False
            try:
                server.help()
            except (OSError, EOFError):
                return False
            return True
        kwargs = dict(timeout=support.INTERNET_TIMEOUT, usenetrc=False)
        if self.ssl_context is not None:
            kwargs['ssl_context'] = self.ssl_context
        try:
            server = self.NNTP_CLASS(self.NNTP_HOST, **kwargs)
            with server:
                self.assertTrue(is_connected())
                self.assertTrue(server.help())
            self.assertFalse(is_connected())
            server = self.NNTP_CLASS(self.NNTP_HOST, **kwargs)
            with server:
                server.quit()
            self.assertFalse(is_connected())
        except SSLError as ssl_err:
            if re.search('(?i)KEY.TOO.SMALL', ssl_err.reason):
                raise unittest.SkipTest(f'Got {ssl_err} connecting to {self.NNTP_HOST!r}')
            raise
NetworkedNNTPTestsMixin.wrap_methods()
EOF_ERRORS = (EOFError,)
if ssl is not None:
    EOF_ERRORS += (ssl.SSLEOFError,)

class NetworkedNNTPTests(NetworkedNNTPTestsMixin, unittest.TestCase):
    NNTP_HOST = 'news.trigofacile.com'
    GROUP_NAME = 'fr.comp.lang.python'
    GROUP_PAT = 'fr.comp.lang.*'
    DESC = 'Python'
    NNTP_CLASS = NNTP

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        support.requires('network')
        kwargs = dict(timeout=support.INTERNET_TIMEOUT, usenetrc=False)
        if cls.ssl_context is not None:
            kwargs['ssl_context'] = cls.ssl_context
        with socket_helper.transient_internet(cls.NNTP_HOST):
            try:
                cls.server = cls.NNTP_CLASS(cls.NNTP_HOST, **kwargs)
            except SSLError as ssl_err:
                if re.search('(?i)KEY.TOO.SMALL', ssl_err.reason):
                    raise unittest.SkipTest(f'{cls} got {ssl_err} connecting to {cls.NNTP_HOST!r}')
                print(cls.NNTP_HOST)
                raise
            except EOF_ERRORS:
                raise unittest.SkipTest(f'{cls} got EOF error on connecting to {cls.NNTP_HOST!r}')

    @classmethod
    def tearDownClass(cls):
        if False:
            print('Hello World!')
        if cls.server is not None:
            cls.server.quit()

@unittest.skip('Does not work well in Cinder CI environment')
@unittest.skipUnless(ssl, 'requires SSL support')
class NetworkedNNTP_SSLTests(NetworkedNNTPTests):
    NNTP_HOST = 'nntp.aioe.org'
    GROUP_NAME = 'aioe.test'
    GROUP_PAT = 'aioe.*'
    DESC = 'test'
    NNTP_CLASS = getattr(nntplib, 'NNTP_SSL', None)
    test_list = None
    test_starttls = None
    if ssl is not None:
        ssl_context = ssl._create_unverified_context()
        ssl_context.set_ciphers('DEFAULT')
        ssl_context.maximum_version = ssl.TLSVersion.TLSv1_2

class _NNTPServerIO(io.RawIOBase):
    """A raw IO object allowing NNTP commands to be received and processed
    by a handler.  The handler can push responses which can then be read
    from the IO object."""

    def __init__(self, handler):
        if False:
            return 10
        io.RawIOBase.__init__(self)
        self.c2s = io.BytesIO()
        self.s2c = io.BytesIO()
        self.handler = handler
        self.handler.start(self.c2s.readline, self.push_data)

    def readable(self):
        if False:
            i = 10
            return i + 15
        return True

    def writable(self):
        if False:
            while True:
                i = 10
        return True

    def push_data(self, data):
        if False:
            while True:
                i = 10
        'Push (buffer) some data to send to the client.'
        pos = self.s2c.tell()
        self.s2c.seek(0, 2)
        self.s2c.write(data)
        self.s2c.seek(pos)

    def write(self, b):
        if False:
            for i in range(10):
                print('nop')
        'The client sends us some data'
        pos = self.c2s.tell()
        self.c2s.write(b)
        self.c2s.seek(pos)
        self.handler.process_pending()
        return len(b)

    def readinto(self, buf):
        if False:
            for i in range(10):
                print('nop')
        'The client wants to read a response'
        self.handler.process_pending()
        b = self.s2c.read(len(buf))
        n = len(b)
        buf[:n] = b
        return n

def make_mock_file(handler):
    if False:
        while True:
            i = 10
    sio = _NNTPServerIO(handler)
    file = io.BufferedRWPair(sio, sio)
    return (sio, file)

class NNTPServer(nntplib.NNTP):

    def __init__(self, f, host, readermode=None):
        if False:
            while True:
                i = 10
        self.file = f
        self.host = host
        self._base_init(readermode)

    def _close(self):
        if False:
            for i in range(10):
                print('nop')
        self.file.close()
        del self.file

class MockedNNTPTestsMixin:
    handler_class = None

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.make_server()

    def tearDown(self):
        if False:
            while True:
                i = 10
        super().tearDown()
        del self.server

    def make_server(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.handler = self.handler_class()
        (self.sio, file) = make_mock_file(self.handler)
        self.server = NNTPServer(file, 'test.server', *args, **kwargs)
        return self.server

class MockedNNTPWithReaderModeMixin(MockedNNTPTestsMixin):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.make_server(readermode=True)

class NNTPv1Handler:
    """A handler for RFC 977"""
    welcome = '200 NNTP mock server'

    def start(self, readline, push_data):
        if False:
            while True:
                i = 10
        self.in_body = False
        self.allow_posting = True
        self._readline = readline
        self._push_data = push_data
        self._logged_in = False
        self._user_sent = False
        self.handle_welcome()

    def _decode(self, data):
        if False:
            i = 10
            return i + 15
        return str(data, 'utf-8', 'surrogateescape')

    def process_pending(self):
        if False:
            for i in range(10):
                print('nop')
        if self.in_body:
            while True:
                line = self._readline()
                if not line:
                    return
                self.body.append(line)
                if line == b'.\r\n':
                    break
            try:
                (meth, tokens) = self.body_callback
                meth(*tokens, body=self.body)
            finally:
                self.body_callback = None
                self.body = None
                self.in_body = False
        while True:
            line = self._decode(self._readline())
            if not line:
                return
            if not line.endswith('\r\n'):
                raise ValueError("line doesn't end with \\r\\n: {!r}".format(line))
            line = line[:-2]
            (cmd, *tokens) = line.split()
            meth = getattr(self, 'handle_' + cmd.upper(), None)
            if meth is None:
                self.handle_unknown()
            else:
                try:
                    meth(*tokens)
                except Exception as e:
                    raise ValueError('command failed: {!r}'.format(line)) from e
                else:
                    if self.in_body:
                        self.body_callback = (meth, tokens)
                        self.body = []

    def expect_body(self):
        if False:
            while True:
                i = 10
        'Flag that the client is expected to post a request body'
        self.in_body = True

    def push_data(self, data):
        if False:
            return 10
        'Push some binary data'
        self._push_data(data)

    def push_lit(self, lit):
        if False:
            while True:
                i = 10
        'Push a string literal'
        lit = textwrap.dedent(lit)
        lit = '\r\n'.join(lit.splitlines()) + '\r\n'
        lit = lit.encode('utf-8')
        self.push_data(lit)

    def handle_unknown(self):
        if False:
            return 10
        self.push_lit('500 What?')

    def handle_welcome(self):
        if False:
            i = 10
            return i + 15
        self.push_lit(self.welcome)

    def handle_QUIT(self):
        if False:
            return 10
        self.push_lit('205 Bye!')

    def handle_DATE(self):
        if False:
            while True:
                i = 10
        self.push_lit('111 20100914001155')

    def handle_GROUP(self, group):
        if False:
            return 10
        if group == 'fr.comp.lang.python':
            self.push_lit('211 486 761 1265 fr.comp.lang.python')
        else:
            self.push_lit('411 No such group {}'.format(group))

    def handle_HELP(self):
        if False:
            while True:
                i = 10
        self.push_lit('            100 Legal commands\n              authinfo user Name|pass Password|generic <prog> <args>\n              date\n              help\n            Report problems to <root@example.org>\n            .')

    def handle_STAT(self, message_spec=None):
        if False:
            return 10
        if message_spec is None:
            self.push_lit('412 No newsgroup selected')
        elif message_spec == '3000234':
            self.push_lit('223 3000234 <45223423@example.com>')
        elif message_spec == '<45223423@example.com>':
            self.push_lit('223 0 <45223423@example.com>')
        else:
            self.push_lit('430 No Such Article Found')

    def handle_NEXT(self):
        if False:
            i = 10
            return i + 15
        self.push_lit('223 3000237 <668929@example.org> retrieved')

    def handle_LAST(self):
        if False:
            print('Hello World!')
        self.push_lit('223 3000234 <45223423@example.com> retrieved')

    def handle_LIST(self, action=None, param=None):
        if False:
            print('Hello World!')
        if action is None:
            self.push_lit('                215 Newsgroups in form "group high low flags".\n                comp.lang.python 0000052340 0000002828 y\n                comp.lang.python.announce 0000001153 0000000993 m\n                free.it.comp.lang.python 0000000002 0000000002 y\n                fr.comp.lang.python 0000001254 0000000760 y\n                free.it.comp.lang.python.learner 0000000000 0000000001 y\n                tw.bbs.comp.lang.python 0000000304 0000000304 y\n                .')
        elif action == 'ACTIVE':
            if param == '*distutils*':
                self.push_lit('                    215 Newsgroups in form "group high low flags"\n                    gmane.comp.python.distutils.devel 0000014104 0000000001 m\n                    gmane.comp.python.distutils.cvs 0000000000 0000000001 m\n                    .')
            else:
                self.push_lit('                    215 Newsgroups in form "group high low flags"\n                    .')
        elif action == 'OVERVIEW.FMT':
            self.push_lit('                215 Order of fields in overview database.\n                Subject:\n                From:\n                Date:\n                Message-ID:\n                References:\n                Bytes:\n                Lines:\n                Xref:full\n                .')
        elif action == 'NEWSGROUPS':
            assert param is not None
            if param == 'comp.lang.python':
                self.push_lit('                    215 Descriptions in form "group description".\n                    comp.lang.python\tThe Python computer language.\n                    .')
            elif param == 'comp.lang.python*':
                self.push_lit('                    215 Descriptions in form "group description".\n                    comp.lang.python.announce\tAnnouncements about the Python language. (Moderated)\n                    comp.lang.python\tThe Python computer language.\n                    .')
            else:
                self.push_lit('                    215 Descriptions in form "group description".\n                    .')
        else:
            self.push_lit('501 Unknown LIST keyword')

    def handle_NEWNEWS(self, group, date_str, time_str):
        if False:
            for i in range(10):
                print('nop')
        if group == 'comp.lang.python' and date_str == '20100913' and (time_str == '082004'):
            self.push_lit('                230 list of newsarticles (NNTP v2) created after Mon Sep 13 08:20:04 2010 follows\n                <a4929a40-6328-491a-aaaf-cb79ed7309a2@q2g2000vbk.googlegroups.com>\n                <f30c0419-f549-4218-848f-d7d0131da931@y3g2000vbm.googlegroups.com>\n                .')
        elif group == 'comp.lang.python' and date_str == '100913' and (time_str == '082004'):
            self.push_lit('                230 list of newsarticles (NNTP v1) created after Mon Sep 13 08:20:04 2010 follows\n                <a4929a40-6328-491a-aaaf-cb79ed7309a2@q2g2000vbk.googlegroups.com>\n                <f30c0419-f549-4218-848f-d7d0131da931@y3g2000vbm.googlegroups.com>\n                .')
        elif group == 'comp.lang.python' and date_str in ('20100101', '100101') and (time_str == '090000'):
            self.push_lit('too long line' * 3000 + '\n.')
        else:
            self.push_lit('                230 An empty list of newsarticles follows\n                .')

    def handle_XOVER(self, message_spec):
        if False:
            i = 10
            return i + 15
        if message_spec == '57-59':
            self.push_lit("224 Overview information for 57-58 follows\n57\tRe: ANN: New Plone book with strong Python (and Zope) themes throughout\tDoug Hellmann <doug.hellmann-Re5JQEeQqe8AvxtiuMwx3w@public.gmane.org>\tSat, 19 Jun 2010 18:04:08 -0400\t<4FD05F05-F98B-44DC-8111-C6009C925F0C@gmail.com>\t<hvalf7$ort$1@dough.gmane.org>\t7103\t16\tXref: news.gmane.io gmane.comp.python.authors:57\n58\tLooking for a few good bloggers\tDoug Hellmann <doug.hellmann-Re5JQEeQqe8AvxtiuMwx3w@public.gmane.org>\tThu, 22 Jul 2010 09:14:14 -0400\t<A29863FA-F388-40C3-AA25-0FD06B09B5BF@gmail.com>\t\t6683\t16\t\n59\tRe: Message d'erreur incompréhensible (par moi)\tEric Brunel <eric.brunel@pragmadev.nospam.com>\tWed, 15 Sep 2010 18:09:15 +0200\t<eric.brunel-2B8B56.18091515092010@news.wanadoo.fr>\t<4c90ec87$0$32425$ba4acef3@reader.news.orange.fr>\t1641\t27\tXref: saria.nerim.net fr.comp.lang.python:1265\n.\n")
        else:
            self.push_lit('                224 No articles\n                .')

    def handle_POST(self, *, body=None):
        if False:
            for i in range(10):
                print('nop')
        if body is None:
            if self.allow_posting:
                self.push_lit('340 Input article; end with <CR-LF>.<CR-LF>')
                self.expect_body()
            else:
                self.push_lit('440 Posting not permitted')
        else:
            assert self.allow_posting
            self.push_lit('240 Article received OK')
            self.posted_body = body

    def handle_IHAVE(self, message_id, *, body=None):
        if False:
            while True:
                i = 10
        if body is None:
            if self.allow_posting and message_id == '<i.am.an.article.you.will.want@example.com>':
                self.push_lit('335 Send it; end with <CR-LF>.<CR-LF>')
                self.expect_body()
            else:
                self.push_lit('435 Article not wanted')
        else:
            assert self.allow_posting
            self.push_lit('235 Article transferred OK')
            self.posted_body = body
    sample_head = '        From: "Demo User" <nobody@example.net>\n        Subject: I am just a test article\n        Content-Type: text/plain; charset=UTF-8; format=flowed\n        Message-ID: <i.am.an.article.you.will.want@example.com>'
    sample_body = '        This is just a test article.\n        ..Here is a dot-starting line.\n\n        -- Signed by André.'
    sample_article = sample_head + '\n\n' + sample_body

    def handle_ARTICLE(self, message_spec=None):
        if False:
            for i in range(10):
                print('nop')
        if message_spec is None:
            self.push_lit('220 3000237 <45223423@example.com>')
        elif message_spec == '<45223423@example.com>':
            self.push_lit('220 0 <45223423@example.com>')
        elif message_spec == '3000234':
            self.push_lit('220 3000234 <45223423@example.com>')
        else:
            self.push_lit('430 No Such Article Found')
            return
        self.push_lit(self.sample_article)
        self.push_lit('.')

    def handle_HEAD(self, message_spec=None):
        if False:
            return 10
        if message_spec is None:
            self.push_lit('221 3000237 <45223423@example.com>')
        elif message_spec == '<45223423@example.com>':
            self.push_lit('221 0 <45223423@example.com>')
        elif message_spec == '3000234':
            self.push_lit('221 3000234 <45223423@example.com>')
        else:
            self.push_lit('430 No Such Article Found')
            return
        self.push_lit(self.sample_head)
        self.push_lit('.')

    def handle_BODY(self, message_spec=None):
        if False:
            return 10
        if message_spec is None:
            self.push_lit('222 3000237 <45223423@example.com>')
        elif message_spec == '<45223423@example.com>':
            self.push_lit('222 0 <45223423@example.com>')
        elif message_spec == '3000234':
            self.push_lit('222 3000234 <45223423@example.com>')
        else:
            self.push_lit('430 No Such Article Found')
            return
        self.push_lit(self.sample_body)
        self.push_lit('.')

    def handle_AUTHINFO(self, cred_type, data):
        if False:
            return 10
        if self._logged_in:
            self.push_lit('502 Already Logged In')
        elif cred_type == 'user':
            if self._user_sent:
                self.push_lit('482 User Credential Already Sent')
            else:
                self.push_lit('381 Password Required')
                self._user_sent = True
        elif cred_type == 'pass':
            self.push_lit('281 Login Successful')
            self._logged_in = True
        else:
            raise Exception('Unknown cred type {}'.format(cred_type))

class NNTPv2Handler(NNTPv1Handler):
    """A handler for RFC 3977 (NNTP "v2")"""

    def handle_CAPABILITIES(self):
        if False:
            print('Hello World!')
        fmt = '            101 Capability list:\n            VERSION 2 3\n            IMPLEMENTATION INN 2.5.1{}\n            HDR\n            LIST ACTIVE ACTIVE.TIMES DISTRIB.PATS HEADERS NEWSGROUPS OVERVIEW.FMT\n            OVER\n            POST\n            READER\n            .'
        if not self._logged_in:
            self.push_lit(fmt.format('\n            AUTHINFO USER'))
        else:
            self.push_lit(fmt.format(''))

    def handle_MODE(self, _):
        if False:
            return 10
        raise Exception('MODE READER sent despite READER has been advertised')

    def handle_OVER(self, message_spec=None):
        if False:
            while True:
                i = 10
        return self.handle_XOVER(message_spec)

class CapsAfterLoginNNTPv2Handler(NNTPv2Handler):
    """A handler that allows CAPABILITIES only after login"""

    def handle_CAPABILITIES(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._logged_in:
            self.push_lit('480 You must log in.')
        else:
            super().handle_CAPABILITIES()

class ModeSwitchingNNTPv2Handler(NNTPv2Handler):
    """A server that starts in transit mode"""

    def __init__(self):
        if False:
            print('Hello World!')
        self._switched = False

    def handle_CAPABILITIES(self):
        if False:
            for i in range(10):
                print('nop')
        fmt = '            101 Capability list:\n            VERSION 2 3\n            IMPLEMENTATION INN 2.5.1\n            HDR\n            LIST ACTIVE ACTIVE.TIMES DISTRIB.PATS HEADERS NEWSGROUPS OVERVIEW.FMT\n            OVER\n            POST\n            {}READER\n            .'
        if self._switched:
            self.push_lit(fmt.format(''))
        else:
            self.push_lit(fmt.format('MODE-'))

    def handle_MODE(self, what):
        if False:
            for i in range(10):
                print('nop')
        assert not self._switched and what == 'reader'
        self._switched = True
        self.push_lit('200 Posting allowed')

class NNTPv1v2TestsMixin:

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()

    def test_welcome(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.server.welcome, self.handler.welcome)

    def test_authinfo(self):
        if False:
            for i in range(10):
                print('nop')
        if self.nntp_version == 2:
            self.assertIn('AUTHINFO', self.server._caps)
        self.server.login('testuser', 'testpw')
        self.assertNotIn('AUTHINFO', self.server._caps)

    def test_date(self):
        if False:
            return 10
        (resp, date) = self.server.date()
        self.assertEqual(resp, '111 20100914001155')
        self.assertEqual(date, datetime.datetime(2010, 9, 14, 0, 11, 55))

    def test_quit(self):
        if False:
            while True:
                i = 10
        self.assertFalse(self.sio.closed)
        resp = self.server.quit()
        self.assertEqual(resp, '205 Bye!')
        self.assertTrue(self.sio.closed)

    def test_help(self):
        if False:
            i = 10
            return i + 15
        (resp, help) = self.server.help()
        self.assertEqual(resp, '100 Legal commands')
        self.assertEqual(help, ['  authinfo user Name|pass Password|generic <prog> <args>', '  date', '  help', 'Report problems to <root@example.org>'])

    def test_list(self):
        if False:
            i = 10
            return i + 15
        (resp, groups) = self.server.list()
        self.assertEqual(len(groups), 6)
        g = groups[1]
        self.assertEqual(g, GroupInfo('comp.lang.python.announce', '0000001153', '0000000993', 'm'))
        (resp, groups) = self.server.list('*distutils*')
        self.assertEqual(len(groups), 2)
        g = groups[0]
        self.assertEqual(g, GroupInfo('gmane.comp.python.distutils.devel', '0000014104', '0000000001', 'm'))

    def test_stat(self):
        if False:
            return 10
        (resp, art_num, message_id) = self.server.stat(3000234)
        self.assertEqual(resp, '223 3000234 <45223423@example.com>')
        self.assertEqual(art_num, 3000234)
        self.assertEqual(message_id, '<45223423@example.com>')
        (resp, art_num, message_id) = self.server.stat('<45223423@example.com>')
        self.assertEqual(resp, '223 0 <45223423@example.com>')
        self.assertEqual(art_num, 0)
        self.assertEqual(message_id, '<45223423@example.com>')
        with self.assertRaises(nntplib.NNTPTemporaryError) as cm:
            self.server.stat('<non.existent.id>')
        self.assertEqual(cm.exception.response, '430 No Such Article Found')
        with self.assertRaises(nntplib.NNTPTemporaryError) as cm:
            self.server.stat()
        self.assertEqual(cm.exception.response, '412 No newsgroup selected')

    def test_next(self):
        if False:
            print('Hello World!')
        (resp, art_num, message_id) = self.server.next()
        self.assertEqual(resp, '223 3000237 <668929@example.org> retrieved')
        self.assertEqual(art_num, 3000237)
        self.assertEqual(message_id, '<668929@example.org>')

    def test_last(self):
        if False:
            while True:
                i = 10
        (resp, art_num, message_id) = self.server.last()
        self.assertEqual(resp, '223 3000234 <45223423@example.com> retrieved')
        self.assertEqual(art_num, 3000234)
        self.assertEqual(message_id, '<45223423@example.com>')

    def test_description(self):
        if False:
            print('Hello World!')
        desc = self.server.description('comp.lang.python')
        self.assertEqual(desc, 'The Python computer language.')
        desc = self.server.description('comp.lang.pythonx')
        self.assertEqual(desc, '')

    def test_descriptions(self):
        if False:
            print('Hello World!')
        (resp, groups) = self.server.descriptions('comp.lang.python')
        self.assertEqual(resp, '215 Descriptions in form "group description".')
        self.assertEqual(groups, {'comp.lang.python': 'The Python computer language.'})
        (resp, groups) = self.server.descriptions('comp.lang.python*')
        self.assertEqual(groups, {'comp.lang.python': 'The Python computer language.', 'comp.lang.python.announce': 'Announcements about the Python language. (Moderated)'})
        (resp, groups) = self.server.descriptions('comp.lang.pythonx')
        self.assertEqual(groups, {})

    def test_group(self):
        if False:
            print('Hello World!')
        (resp, count, first, last, group) = self.server.group('fr.comp.lang.python')
        self.assertTrue(resp.startswith('211 '), resp)
        self.assertEqual(first, 761)
        self.assertEqual(last, 1265)
        self.assertEqual(count, 486)
        self.assertEqual(group, 'fr.comp.lang.python')
        with self.assertRaises(nntplib.NNTPTemporaryError) as cm:
            self.server.group('comp.lang.python.devel')
        exc = cm.exception
        self.assertTrue(exc.response.startswith('411 No such group'), exc.response)

    def test_newnews(self):
        if False:
            i = 10
            return i + 15
        dt = datetime.datetime(2010, 9, 13, 8, 20, 4)
        (resp, ids) = self.server.newnews('comp.lang.python', dt)
        expected = '230 list of newsarticles (NNTP v{0}) created after Mon Sep 13 08:20:04 2010 follows'.format(self.nntp_version)
        self.assertEqual(resp, expected)
        self.assertEqual(ids, ['<a4929a40-6328-491a-aaaf-cb79ed7309a2@q2g2000vbk.googlegroups.com>', '<f30c0419-f549-4218-848f-d7d0131da931@y3g2000vbm.googlegroups.com>'])
        dt = datetime.datetime(2010, 9, 13, 8, 20, 4)
        (resp, ids) = self.server.newnews('fr.comp.lang.python', dt)
        self.assertEqual(resp, '230 An empty list of newsarticles follows')
        self.assertEqual(ids, [])

    def _check_article_body(self, lines):
        if False:
            return 10
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[-1].decode('utf-8'), '-- Signed by André.')
        self.assertEqual(lines[-2], b'')
        self.assertEqual(lines[-3], b'.Here is a dot-starting line.')
        self.assertEqual(lines[-4], b'This is just a test article.')

    def _check_article_head(self, lines):
        if False:
            while True:
                i = 10
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], b'From: "Demo User" <nobody@example.net>')
        self.assertEqual(lines[3], b'Message-ID: <i.am.an.article.you.will.want@example.com>')

    def _check_article_data(self, lines):
        if False:
            print('Hello World!')
        self.assertEqual(len(lines), 9)
        self._check_article_head(lines[:4])
        self._check_article_body(lines[-4:])
        self.assertEqual(lines[4], b'')

    def test_article(self):
        if False:
            i = 10
            return i + 15
        (resp, info) = self.server.article()
        self.assertEqual(resp, '220 3000237 <45223423@example.com>')
        (art_num, message_id, lines) = info
        self.assertEqual(art_num, 3000237)
        self.assertEqual(message_id, '<45223423@example.com>')
        self._check_article_data(lines)
        (resp, info) = self.server.article(3000234)
        self.assertEqual(resp, '220 3000234 <45223423@example.com>')
        (art_num, message_id, lines) = info
        self.assertEqual(art_num, 3000234)
        self.assertEqual(message_id, '<45223423@example.com>')
        self._check_article_data(lines)
        (resp, info) = self.server.article('<45223423@example.com>')
        self.assertEqual(resp, '220 0 <45223423@example.com>')
        (art_num, message_id, lines) = info
        self.assertEqual(art_num, 0)
        self.assertEqual(message_id, '<45223423@example.com>')
        self._check_article_data(lines)
        with self.assertRaises(nntplib.NNTPTemporaryError) as cm:
            self.server.article('<non-existent@example.com>')
        self.assertEqual(cm.exception.response, '430 No Such Article Found')

    def test_article_file(self):
        if False:
            for i in range(10):
                print('nop')
        f = io.BytesIO()
        (resp, info) = self.server.article(file=f)
        self.assertEqual(resp, '220 3000237 <45223423@example.com>')
        (art_num, message_id, lines) = info
        self.assertEqual(art_num, 3000237)
        self.assertEqual(message_id, '<45223423@example.com>')
        self.assertEqual(lines, [])
        data = f.getvalue()
        self.assertTrue(data.startswith(b'From: "Demo User" <nobody@example.net>\r\nSubject: I am just a test article\r\n'), ascii(data))
        self.assertTrue(data.endswith(b'This is just a test article.\r\n.Here is a dot-starting line.\r\n\r\n-- Signed by Andr\xc3\xa9.\r\n'), ascii(data))

    def test_head(self):
        if False:
            print('Hello World!')
        (resp, info) = self.server.head()
        self.assertEqual(resp, '221 3000237 <45223423@example.com>')
        (art_num, message_id, lines) = info
        self.assertEqual(art_num, 3000237)
        self.assertEqual(message_id, '<45223423@example.com>')
        self._check_article_head(lines)
        (resp, info) = self.server.head(3000234)
        self.assertEqual(resp, '221 3000234 <45223423@example.com>')
        (art_num, message_id, lines) = info
        self.assertEqual(art_num, 3000234)
        self.assertEqual(message_id, '<45223423@example.com>')
        self._check_article_head(lines)
        (resp, info) = self.server.head('<45223423@example.com>')
        self.assertEqual(resp, '221 0 <45223423@example.com>')
        (art_num, message_id, lines) = info
        self.assertEqual(art_num, 0)
        self.assertEqual(message_id, '<45223423@example.com>')
        self._check_article_head(lines)
        with self.assertRaises(nntplib.NNTPTemporaryError) as cm:
            self.server.head('<non-existent@example.com>')
        self.assertEqual(cm.exception.response, '430 No Such Article Found')

    def test_head_file(self):
        if False:
            while True:
                i = 10
        f = io.BytesIO()
        (resp, info) = self.server.head(file=f)
        self.assertEqual(resp, '221 3000237 <45223423@example.com>')
        (art_num, message_id, lines) = info
        self.assertEqual(art_num, 3000237)
        self.assertEqual(message_id, '<45223423@example.com>')
        self.assertEqual(lines, [])
        data = f.getvalue()
        self.assertTrue(data.startswith(b'From: "Demo User" <nobody@example.net>\r\nSubject: I am just a test article\r\n'), ascii(data))
        self.assertFalse(data.endswith(b'This is just a test article.\r\n.Here is a dot-starting line.\r\n\r\n-- Signed by Andr\xc3\xa9.\r\n'), ascii(data))

    def test_body(self):
        if False:
            i = 10
            return i + 15
        (resp, info) = self.server.body()
        self.assertEqual(resp, '222 3000237 <45223423@example.com>')
        (art_num, message_id, lines) = info
        self.assertEqual(art_num, 3000237)
        self.assertEqual(message_id, '<45223423@example.com>')
        self._check_article_body(lines)
        (resp, info) = self.server.body(3000234)
        self.assertEqual(resp, '222 3000234 <45223423@example.com>')
        (art_num, message_id, lines) = info
        self.assertEqual(art_num, 3000234)
        self.assertEqual(message_id, '<45223423@example.com>')
        self._check_article_body(lines)
        (resp, info) = self.server.body('<45223423@example.com>')
        self.assertEqual(resp, '222 0 <45223423@example.com>')
        (art_num, message_id, lines) = info
        self.assertEqual(art_num, 0)
        self.assertEqual(message_id, '<45223423@example.com>')
        self._check_article_body(lines)
        with self.assertRaises(nntplib.NNTPTemporaryError) as cm:
            self.server.body('<non-existent@example.com>')
        self.assertEqual(cm.exception.response, '430 No Such Article Found')

    def test_body_file(self):
        if False:
            return 10
        f = io.BytesIO()
        (resp, info) = self.server.body(file=f)
        self.assertEqual(resp, '222 3000237 <45223423@example.com>')
        (art_num, message_id, lines) = info
        self.assertEqual(art_num, 3000237)
        self.assertEqual(message_id, '<45223423@example.com>')
        self.assertEqual(lines, [])
        data = f.getvalue()
        self.assertFalse(data.startswith(b'From: "Demo User" <nobody@example.net>\r\nSubject: I am just a test article\r\n'), ascii(data))
        self.assertTrue(data.endswith(b'This is just a test article.\r\n.Here is a dot-starting line.\r\n\r\n-- Signed by Andr\xc3\xa9.\r\n'), ascii(data))

    def check_over_xover_resp(self, resp, overviews):
        if False:
            while True:
                i = 10
        self.assertTrue(resp.startswith('224 '), resp)
        self.assertEqual(len(overviews), 3)
        (art_num, over) = overviews[0]
        self.assertEqual(art_num, 57)
        self.assertEqual(over, {'from': 'Doug Hellmann <doug.hellmann-Re5JQEeQqe8AvxtiuMwx3w@public.gmane.org>', 'subject': 'Re: ANN: New Plone book with strong Python (and Zope) themes throughout', 'date': 'Sat, 19 Jun 2010 18:04:08 -0400', 'message-id': '<4FD05F05-F98B-44DC-8111-C6009C925F0C@gmail.com>', 'references': '<hvalf7$ort$1@dough.gmane.org>', ':bytes': '7103', ':lines': '16', 'xref': 'news.gmane.io gmane.comp.python.authors:57'})
        (art_num, over) = overviews[1]
        self.assertEqual(over['xref'], None)
        (art_num, over) = overviews[2]
        self.assertEqual(over['subject'], "Re: Message d'erreur incompréhensible (par moi)")

    def test_xover(self):
        if False:
            i = 10
            return i + 15
        (resp, overviews) = self.server.xover(57, 59)
        self.check_over_xover_resp(resp, overviews)

    def test_over(self):
        if False:
            for i in range(10):
                print('nop')
        (resp, overviews) = self.server.over((57, 59))
        self.check_over_xover_resp(resp, overviews)
    sample_post = b'From: "Demo User" <nobody@example.net>\r\nSubject: I am just a test article\r\nContent-Type: text/plain; charset=UTF-8; format=flowed\r\nMessage-ID: <i.am.an.article.you.will.want@example.com>\r\n\r\nThis is just a test article.\r\n.Here is a dot-starting line.\r\n\r\n-- Signed by Andr\xc3\xa9.\r\n'

    def _check_posted_body(self):
        if False:
            for i in range(10):
                print('nop')
        lines = self.handler.posted_body
        self.assertEqual(len(lines), 10)
        self.assertEqual(lines[-1], b'.\r\n')
        self.assertEqual(lines[-2], b'-- Signed by Andr\xc3\xa9.\r\n')
        self.assertEqual(lines[-3], b'\r\n')
        self.assertEqual(lines[-4], b'..Here is a dot-starting line.\r\n')
        self.assertEqual(lines[0], b'From: "Demo User" <nobody@example.net>\r\n')

    def _check_post_ihave_sub(self, func, *args, file_factory):
        if False:
            while True:
                i = 10
        post = self.sample_post
        func_args = args + (file_factory(post),)
        self.handler.posted_body = None
        resp = func(*func_args)
        self._check_posted_body()
        post = self.sample_post.replace(b'\r\n', b'\n')
        func_args = args + (file_factory(post),)
        self.handler.posted_body = None
        resp = func(*func_args)
        self._check_posted_body()
        return resp

    def check_post_ihave(self, func, success_resp, *args):
        if False:
            i = 10
            return i + 15
        resp = self._check_post_ihave_sub(func, *args, file_factory=bytes)
        self.assertEqual(resp, success_resp)
        resp = self._check_post_ihave_sub(func, *args, file_factory=bytearray)
        self.assertEqual(resp, success_resp)
        resp = self._check_post_ihave_sub(func, *args, file_factory=io.BytesIO)
        self.assertEqual(resp, success_resp)

        def iterlines(b):
            if False:
                return 10
            return iter(b.splitlines(keepends=True))
        resp = self._check_post_ihave_sub(func, *args, file_factory=iterlines)
        self.assertEqual(resp, success_resp)

        def iterlines(b):
            if False:
                for i in range(10):
                    print('nop')
            return iter(b.splitlines(keepends=False))
        resp = self._check_post_ihave_sub(func, *args, file_factory=iterlines)
        self.assertEqual(resp, success_resp)

    def test_post(self):
        if False:
            while True:
                i = 10
        self.check_post_ihave(self.server.post, '240 Article received OK')
        self.handler.allow_posting = False
        with self.assertRaises(nntplib.NNTPTemporaryError) as cm:
            self.server.post(self.sample_post)
        self.assertEqual(cm.exception.response, '440 Posting not permitted')

    def test_ihave(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_post_ihave(self.server.ihave, '235 Article transferred OK', '<i.am.an.article.you.will.want@example.com>')
        with self.assertRaises(nntplib.NNTPTemporaryError) as cm:
            self.server.ihave('<another.message.id>', self.sample_post)
        self.assertEqual(cm.exception.response, '435 Article not wanted')

    def test_too_long_lines(self):
        if False:
            print('Hello World!')
        dt = datetime.datetime(2010, 1, 1, 9, 0, 0)
        self.assertRaises(nntplib.NNTPDataError, self.server.newnews, 'comp.lang.python', dt)

class NNTPv1Tests(NNTPv1v2TestsMixin, MockedNNTPTestsMixin, unittest.TestCase):
    """Tests an NNTP v1 server (no capabilities)."""
    nntp_version = 1
    handler_class = NNTPv1Handler

    def test_caps(self):
        if False:
            return 10
        caps = self.server.getcapabilities()
        self.assertEqual(caps, {})
        self.assertEqual(self.server.nntp_version, 1)
        self.assertEqual(self.server.nntp_implementation, None)

class NNTPv2Tests(NNTPv1v2TestsMixin, MockedNNTPTestsMixin, unittest.TestCase):
    """Tests an NNTP v2 server (with capabilities)."""
    nntp_version = 2
    handler_class = NNTPv2Handler

    def test_caps(self):
        if False:
            i = 10
            return i + 15
        caps = self.server.getcapabilities()
        self.assertEqual(caps, {'VERSION': ['2', '3'], 'IMPLEMENTATION': ['INN', '2.5.1'], 'AUTHINFO': ['USER'], 'HDR': [], 'LIST': ['ACTIVE', 'ACTIVE.TIMES', 'DISTRIB.PATS', 'HEADERS', 'NEWSGROUPS', 'OVERVIEW.FMT'], 'OVER': [], 'POST': [], 'READER': []})
        self.assertEqual(self.server.nntp_version, 3)
        self.assertEqual(self.server.nntp_implementation, 'INN 2.5.1')

class CapsAfterLoginNNTPv2Tests(MockedNNTPTestsMixin, unittest.TestCase):
    """Tests a probably NNTP v2 server with capabilities only after login."""
    nntp_version = 2
    handler_class = CapsAfterLoginNNTPv2Handler

    def test_caps_only_after_login(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.server._caps, {})
        self.server.login('testuser', 'testpw')
        self.assertIn('VERSION', self.server._caps)

class SendReaderNNTPv2Tests(MockedNNTPWithReaderModeMixin, unittest.TestCase):
    """Same tests as for v2 but we tell NTTP to send MODE READER to a server
    that isn't in READER mode by default."""
    nntp_version = 2
    handler_class = ModeSwitchingNNTPv2Handler

    def test_we_are_in_reader_mode_after_connect(self):
        if False:
            print('Hello World!')
        self.assertIn('READER', self.server._caps)

class MiscTests(unittest.TestCase):

    def test_decode_header(self):
        if False:
            return 10

        def gives(a, b):
            if False:
                i = 10
                return i + 15
            self.assertEqual(nntplib.decode_header(a), b)
        gives('', '')
        gives('a plain header', 'a plain header')
        gives(' with extra  spaces ', ' with extra  spaces ')
        gives('=?ISO-8859-15?Q?D=E9buter_en_Python?=', 'Débuter en Python')
        gives('=?utf-8?q?Re=3A_=5Bsqlite=5D_probl=C3=A8me_avec_ORDER_BY_sur_des_cha?= =?utf-8?q?=C3=AEnes_de_caract=C3=A8res_accentu=C3=A9es?=', 'Re: [sqlite] problème avec ORDER BY sur des chaînes de caractères accentuées')
        gives('Re: =?UTF-8?B?cHJvYmzDqG1lIGRlIG1hdHJpY2U=?=', 'Re: problème de matrice')
        gives("Re: Message d'erreur incompréhensible (par moi)", "Re: Message d'erreur incompréhensible (par moi)")

    def test_parse_overview_fmt(self):
        if False:
            i = 10
            return i + 15
        lines = ['Subject:', 'From:', 'Date:', 'Message-ID:', 'References:', ':bytes', ':lines']
        self.assertEqual(nntplib._parse_overview_fmt(lines), ['subject', 'from', 'date', 'message-id', 'references', ':bytes', ':lines'])
        lines = ['Subject:', 'From:', 'Date:', 'Message-ID:', 'References:', 'Bytes:', 'Lines:']
        self.assertEqual(nntplib._parse_overview_fmt(lines), ['subject', 'from', 'date', 'message-id', 'references', ':bytes', ':lines'])
        lines = ['subject:', 'FROM:', 'DaTe:', 'message-ID:', 'References:', 'BYTES:', 'Lines:']
        self.assertEqual(nntplib._parse_overview_fmt(lines), ['subject', 'from', 'date', 'message-id', 'references', ':bytes', ':lines'])
        lines = ['Subject:', 'From:', 'Date:', 'Message-ID:', 'References:', ':bytes', ':lines', 'Xref:full', 'Distribution:full']
        self.assertEqual(nntplib._parse_overview_fmt(lines), ['subject', 'from', 'date', 'message-id', 'references', ':bytes', ':lines', 'xref', 'distribution'])
        lines = ['Subject:', 'From:', 'Date:', 'Message-ID:', 'References:', 'Bytes:', 'Lines:', 'Xref:FULL', 'Distribution:FULL']
        self.assertEqual(nntplib._parse_overview_fmt(lines), ['subject', 'from', 'date', 'message-id', 'references', ':bytes', ':lines', 'xref', 'distribution'])
        lines = ['Subject:', 'From:', 'Date:', 'Message-ID:', 'References:', 'Bytes:', 'Lines:', 'Xref:full']
        self.assertEqual(nntplib._parse_overview_fmt(lines), ['subject', 'from', 'date', 'message-id', 'references', ':bytes', ':lines', 'xref'])

    def test_parse_overview(self):
        if False:
            print('Hello World!')
        fmt = nntplib._DEFAULT_OVERVIEW_FMT + ['xref']
        lines = ['3000234\tI am just a test article\t"Demo User" <nobody@example.com>\t6 Oct 1998 04:38:40 -0500\t<45223423@example.com>\t<45454@example.net>\t1234\t17\tXref: news.example.com misc.test:3000363']
        overview = nntplib._parse_overview(lines, fmt)
        ((art_num, fields),) = overview
        self.assertEqual(art_num, 3000234)
        self.assertEqual(fields, {'subject': 'I am just a test article', 'from': '"Demo User" <nobody@example.com>', 'date': '6 Oct 1998 04:38:40 -0500', 'message-id': '<45223423@example.com>', 'references': '<45454@example.net>', ':bytes': '1234', ':lines': '17', 'xref': 'news.example.com misc.test:3000363'})
        lines = ['3000234\tI am just a test article\t"Demo User" <nobody@example.com>\t6 Oct 1998 04:38:40 -0500\t<45223423@example.com>\t<45454@example.net>\t1234\t17\t\t']
        overview = nntplib._parse_overview(lines, fmt)
        ((art_num, fields),) = overview
        self.assertEqual(fields['xref'], None)
        lines = ['3000234\tI am just a test article\t"Demo User" <nobody@example.com>\t6 Oct 1998 04:38:40 -0500\t<45223423@example.com>\t \t1234\t17\tXref: \t']
        overview = nntplib._parse_overview(lines, fmt)
        ((art_num, fields),) = overview
        self.assertEqual(fields['references'], ' ')
        self.assertEqual(fields['xref'], '')

    def test_parse_datetime(self):
        if False:
            return 10

        def gives(a, b, *c):
            if False:
                while True:
                    i = 10
            self.assertEqual(nntplib._parse_datetime(a, b), datetime.datetime(*c))
        gives('19990623135624', None, 1999, 6, 23, 13, 56, 24)
        gives('19990623', '135624', 1999, 6, 23, 13, 56, 24)
        gives('990623', '135624', 1999, 6, 23, 13, 56, 24)
        gives('090623', '135624', 2009, 6, 23, 13, 56, 24)

    def test_unparse_datetime(self):
        if False:
            i = 10
            return i + 15

        def gives(y, M, d, h, m, s, date_str, time_str):
            if False:
                i = 10
                return i + 15
            dt = datetime.datetime(y, M, d, h, m, s)
            self.assertEqual(nntplib._unparse_datetime(dt), (date_str, time_str))
            self.assertEqual(nntplib._unparse_datetime(dt, False), (date_str, time_str))
        gives(1999, 6, 23, 13, 56, 24, '19990623', '135624')
        gives(2000, 6, 23, 13, 56, 24, '20000623', '135624')
        gives(2010, 6, 5, 1, 2, 3, '20100605', '010203')

        def gives(y, M, d, date_str, time_str):
            if False:
                i = 10
                return i + 15
            dt = datetime.date(y, M, d)
            self.assertEqual(nntplib._unparse_datetime(dt), (date_str, time_str))
            self.assertEqual(nntplib._unparse_datetime(dt, False), (date_str, time_str))
        gives(1999, 6, 23, '19990623', '000000')
        gives(2000, 6, 23, '20000623', '000000')
        gives(2010, 6, 5, '20100605', '000000')

    def test_unparse_datetime_legacy(self):
        if False:
            return 10

        def gives(y, M, d, h, m, s, date_str, time_str):
            if False:
                while True:
                    i = 10
            dt = datetime.datetime(y, M, d, h, m, s)
            self.assertEqual(nntplib._unparse_datetime(dt, True), (date_str, time_str))
        gives(1999, 6, 23, 13, 56, 24, '990623', '135624')
        gives(2000, 6, 23, 13, 56, 24, '000623', '135624')
        gives(2010, 6, 5, 1, 2, 3, '100605', '010203')

        def gives(y, M, d, date_str, time_str):
            if False:
                i = 10
                return i + 15
            dt = datetime.date(y, M, d)
            self.assertEqual(nntplib._unparse_datetime(dt, True), (date_str, time_str))
        gives(1999, 6, 23, '990623', '000000')
        gives(2000, 6, 23, '000623', '000000')
        gives(2010, 6, 5, '100605', '000000')

    @unittest.skipUnless(ssl, 'requires SSL support')
    def test_ssl_support(self):
        if False:
            print('Hello World!')
        self.assertTrue(hasattr(nntplib, 'NNTP_SSL'))

class PublicAPITests(unittest.TestCase):
    """Ensures that the correct values are exposed in the public API."""

    def test_module_all_attribute(self):
        if False:
            while True:
                i = 10
        self.assertTrue(hasattr(nntplib, '__all__'))
        target_api = ['NNTP', 'NNTPError', 'NNTPReplyError', 'NNTPTemporaryError', 'NNTPPermanentError', 'NNTPProtocolError', 'NNTPDataError', 'decode_header']
        if ssl is not None:
            target_api.append('NNTP_SSL')
        self.assertEqual(set(nntplib.__all__), set(target_api))

class MockSocketTests(unittest.TestCase):
    """Tests involving a mock socket object

    Used where the _NNTPServerIO file object is not enough."""
    nntp_class = nntplib.NNTP

    def check_constructor_error_conditions(self, handler_class, expected_error_type, expected_error_msg, login=None, password=None):
        if False:
            for i in range(10):
                print('nop')

        class mock_socket_module:

            def create_connection(address, timeout):
                if False:
                    for i in range(10):
                        print('nop')
                return MockSocket()

        class MockSocket:

            def close(self):
                if False:
                    while True:
                        i = 10
                nonlocal socket_closed
                socket_closed = True

            def makefile(socket, mode):
                if False:
                    i = 10
                    return i + 15
                handler = handler_class()
                (_, file) = make_mock_file(handler)
                files.append(file)
                return file
        socket_closed = False
        files = []
        with patch('nntplib.socket', mock_socket_module), self.assertRaisesRegex(expected_error_type, expected_error_msg):
            self.nntp_class('dummy', user=login, password=password)
        self.assertTrue(socket_closed)
        for f in files:
            self.assertTrue(f.closed)

    def test_bad_welcome(self):
        if False:
            while True:
                i = 10

        class Handler(NNTPv1Handler):
            welcome = 'Bad Welcome'
        self.check_constructor_error_conditions(Handler, nntplib.NNTPProtocolError, Handler.welcome)

    def test_service_temporarily_unavailable(self):
        if False:
            i = 10
            return i + 15

        class Handler(NNTPv1Handler):
            welcome = '400 Service temporarily unavailable'
        self.check_constructor_error_conditions(Handler, nntplib.NNTPTemporaryError, Handler.welcome)

    def test_service_permanently_unavailable(self):
        if False:
            return 10

        class Handler(NNTPv1Handler):
            welcome = '502 Service permanently unavailable'
        self.check_constructor_error_conditions(Handler, nntplib.NNTPPermanentError, Handler.welcome)

    def test_bad_capabilities(self):
        if False:
            i = 10
            return i + 15

        class Handler(NNTPv1Handler):

            def handle_CAPABILITIES(self):
                if False:
                    return 10
                self.push_lit(capabilities_response)
        capabilities_response = '201 bad capability'
        self.check_constructor_error_conditions(Handler, nntplib.NNTPReplyError, capabilities_response)

    def test_login_aborted(self):
        if False:
            while True:
                i = 10
        login = 't@e.com'
        password = 'python'

        class Handler(NNTPv1Handler):

            def handle_AUTHINFO(self, *args):
                if False:
                    while True:
                        i = 10
                self.push_lit(authinfo_response)
        authinfo_response = '503 Mechanism not recognized'
        self.check_constructor_error_conditions(Handler, nntplib.NNTPPermanentError, authinfo_response, login, password)

class bypass_context:
    """Bypass encryption and actual SSL module"""

    def wrap_socket(sock, **args):
        if False:
            print('Hello World!')
        return sock

@unittest.skipUnless(ssl, 'requires SSL support')
class MockSslTests(MockSocketTests):

    @staticmethod
    def nntp_class(*pos, **kw):
        if False:
            for i in range(10):
                print('nop')
        return nntplib.NNTP_SSL(*pos, ssl_context=bypass_context, **kw)

class LocalServerTests(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        sock = socket.socket()
        port = socket_helper.bind_port(sock)
        sock.listen()
        self.background = threading.Thread(target=self.run_server, args=(sock,))
        self.background.start()
        self.addCleanup(self.background.join)
        self.nntp = NNTP(socket_helper.HOST, port, usenetrc=False).__enter__()
        self.addCleanup(self.nntp.__exit__, None, None, None)

    def run_server(self, sock):
        if False:
            return 10
        with sock:
            [client, _] = sock.accept()
        with contextlib.ExitStack() as cleanup:
            cleanup.enter_context(client)
            reader = cleanup.enter_context(client.makefile('rb'))
            client.sendall(b'200 Server ready\r\n')
            while True:
                cmd = reader.readline()
                if cmd == b'CAPABILITIES\r\n':
                    client.sendall(b'101 Capability list:\r\nVERSION 2\r\nSTARTTLS\r\n.\r\n')
                elif cmd == b'STARTTLS\r\n':
                    reader.close()
                    client.sendall(b'382 Begin TLS negotiation now\r\n')
                    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                    context.load_cert_chain(certfile)
                    client = context.wrap_socket(client, server_side=True)
                    cleanup.enter_context(client)
                    reader = cleanup.enter_context(client.makefile('rb'))
                elif cmd == b'QUIT\r\n':
                    client.sendall(b'205 Bye!\r\n')
                    break
                else:
                    raise ValueError('Unexpected command {!r}'.format(cmd))

    @unittest.skipUnless(ssl, 'requires SSL support')
    def test_starttls(self):
        if False:
            for i in range(10):
                print('nop')
        file = self.nntp.file
        sock = self.nntp.sock
        self.nntp.starttls()
        self.assertNotEqual(file, self.nntp.file)
        self.assertNotEqual(sock, self.nntp.sock)
        self.assertIsInstance(self.nntp.sock, ssl.SSLSocket)
        self.assertRaises(ValueError, self.nntp.starttls)
if __name__ == '__main__':
    unittest.main()