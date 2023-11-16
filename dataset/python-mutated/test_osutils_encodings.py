"""Tests for the osutils wrapper."""
import codecs
import locale
import sys
from bzrlib import osutils
from bzrlib.tests import StringIOWrapper, TestCase

class FakeCodec(object):
    """Special class that helps testing over several non-existed encodings.

    Clients can add new encoding names, but because of how codecs is
    implemented they cannot be removed. Be careful with naming to avoid
    collisions between tests.
    """
    _registered = False
    _enabled_encodings = set()

    def add(self, encoding_name):
        if False:
            print('Hello World!')
        'Adding encoding name to fake.\n\n        :type   encoding_name:  lowercase plain string\n        '
        if not self._registered:
            codecs.register(self)
            self._registered = True
        if encoding_name is not None:
            self._enabled_encodings.add(encoding_name)

    def __call__(self, encoding_name):
        if False:
            for i in range(10):
                print('nop')
        'Called indirectly by codecs module during lookup'
        if encoding_name in self._enabled_encodings:
            return codecs.lookup('latin-1')
fake_codec = FakeCodec()

class TestFakeCodec(TestCase):

    def test_fake_codec(self):
        if False:
            print('Hello World!')
        self.assertRaises(LookupError, codecs.lookup, 'fake')
        fake_codec.add('fake')
        codecs.lookup('fake')

class TestTerminalEncoding(TestCase):
    """Test the auto-detection of proper terminal encoding."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestTerminalEncoding, self).setUp()
        self.overrideAttr(sys, 'stdin')
        self.overrideAttr(sys, 'stdout')
        self.overrideAttr(sys, 'stderr')
        self.overrideAttr(osutils, '_cached_user_encoding')

    def make_wrapped_streams(self, stdout_encoding, stderr_encoding, stdin_encoding, user_encoding='user_encoding', enable_fake_encodings=True):
        if False:
            return 10
        sys.stdout = StringIOWrapper()
        sys.stdout.encoding = stdout_encoding
        sys.stderr = StringIOWrapper()
        sys.stderr.encoding = stderr_encoding
        sys.stdin = StringIOWrapper()
        sys.stdin.encoding = stdin_encoding
        osutils._cached_user_encoding = user_encoding
        if enable_fake_encodings:
            fake_codec.add(stdout_encoding)
            fake_codec.add(stderr_encoding)
            fake_codec.add(stdin_encoding)

    def test_get_terminal_encoding(self):
        if False:
            return 10
        self.make_wrapped_streams('stdout_encoding', 'stderr_encoding', 'stdin_encoding')
        self.assertEqual('stdout_encoding', osutils.get_terminal_encoding())
        sys.stdout.encoding = None
        self.assertEqual('stdin_encoding', osutils.get_terminal_encoding())
        sys.stdin.encoding = None
        self.assertEqual('user_encoding', osutils.get_terminal_encoding())

    def test_get_terminal_encoding_silent(self):
        if False:
            for i in range(10):
                print('nop')
        self.make_wrapped_streams('stdout_encoding', 'stderr_encoding', 'stdin_encoding')
        log = self.get_log()
        osutils.get_terminal_encoding()
        self.assertEqual(log, self.get_log())

    def test_get_terminal_encoding_trace(self):
        if False:
            print('Hello World!')
        self.make_wrapped_streams('stdout_encoding', 'stderr_encoding', 'stdin_encoding')
        log = self.get_log()
        osutils.get_terminal_encoding(trace=True)
        self.assertNotEqual(log, self.get_log())

    def test_terminal_cp0(self):
        if False:
            print('Hello World!')
        self.make_wrapped_streams('cp0', 'cp0', 'cp0', user_encoding='latin-1', enable_fake_encodings=False)
        self.assertEqual('latin-1', osutils.get_terminal_encoding())
        self.assertEqual('', sys.stderr.getvalue())

    def test_terminal_cp_unknown(self):
        if False:
            i = 10
            return i + 15
        self.make_wrapped_streams('cp-unknown', 'cp-unknown', 'cp-unknown', user_encoding='latin-1', enable_fake_encodings=False)
        self.assertEqual('latin-1', osutils.get_terminal_encoding())
        self.assertEqual('bzr: warning: unknown terminal encoding cp-unknown.\n  Using encoding latin-1 instead.\n', sys.stderr.getvalue())

class TestUserEncoding(TestCase):
    """Test detection of default user encoding."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(TestUserEncoding, self).setUp()
        self.overrideAttr(osutils, '_cached_user_encoding', None)
        self.overrideAttr(locale, 'getpreferredencoding', self.get_encoding)
        self.overrideAttr(locale, 'CODESET', None)
        self.overrideAttr(sys, 'stderr', StringIOWrapper())

    def get_encoding(self, do_setlocale=True):
        if False:
            return 10
        return self._encoding

    def test_get_user_encoding(self):
        if False:
            i = 10
            return i + 15
        self._encoding = 'user_encoding'
        fake_codec.add('user_encoding')
        self.assertEqual('iso8859-1', osutils.get_user_encoding())
        self.assertEqual('', sys.stderr.getvalue())

    def test_user_cp0(self):
        if False:
            while True:
                i = 10
        self._encoding = 'cp0'
        self.assertEqual('ascii', osutils.get_user_encoding())
        self.assertEqual('', sys.stderr.getvalue())

    def test_user_cp_unknown(self):
        if False:
            i = 10
            return i + 15
        self._encoding = 'cp-unknown'
        self.assertEqual('ascii', osutils.get_user_encoding())
        self.assertEqual('bzr: warning: unknown encoding cp-unknown. Continuing with ascii encoding.\n', sys.stderr.getvalue())

    def test_user_empty(self):
        if False:
            for i in range(10):
                print('nop')
        "Running bzr from a vim script gives '' for a preferred locale"
        self._encoding = ''
        self.assertEqual('ascii', osutils.get_user_encoding())
        self.assertEqual('', sys.stderr.getvalue())