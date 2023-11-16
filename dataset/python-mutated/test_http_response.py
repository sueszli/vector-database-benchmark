"""Tests from HTTP response parsing.

The handle_response method read the response body of a GET request an returns
the corresponding RangeFile.

There are four different kinds of RangeFile:
- a whole file whose size is unknown, seen as a simple byte stream,
- a whole file whose size is known, we can't read past its end,
- a single range file, a part of a file with a start and a size,
- a multiple range file, several consecutive parts with known start offset
  and size.

Some properties are common to all kinds:
- seek can only be forward (its really a socket underneath),
- read can't cross ranges,
- successive ranges are taken into account transparently,

- the expected pattern of use is either seek(offset)+read(size) or a single
  read with no size specified. For multiple range files, multiple read() will
  return the corresponding ranges, trying to read further will raise
  InvalidHttpResponse.
"""
from cStringIO import StringIO
import httplib
from bzrlib import errors, tests
from bzrlib.transport.http import response, _urllib2_wrappers
from bzrlib.tests.file_utils import FakeReadFile

class ReadSocket(object):
    """A socket-like object that can be given a predefined content."""

    def __init__(self, data):
        if False:
            while True:
                i = 10
        self.readfile = StringIO(data)

    def makefile(self, mode='r', bufsize=None):
        if False:
            return 10
        return self.readfile

class FakeHTTPConnection(_urllib2_wrappers.HTTPConnection):

    def __init__(self, sock):
        if False:
            i = 10
            return i + 15
        _urllib2_wrappers.HTTPConnection.__init__(self, 'localhost')
        self.sock = sock

    def send(self, str):
        if False:
            return 10
        'Ignores the writes on the socket.'
        pass

class TestResponseFileIter(tests.TestCase):

    def test_iter_empty(self):
        if False:
            print('Hello World!')
        f = response.ResponseFile('empty', StringIO())
        self.assertEqual([], list(f))

    def test_iter_many(self):
        if False:
            print('Hello World!')
        f = response.ResponseFile('many', StringIO('0\n1\nboo!\n'))
        self.assertEqual(['0\n', '1\n', 'boo!\n'], list(f))

class TestHTTPConnection(tests.TestCase):

    def test_cleanup_pipe(self):
        if False:
            for i in range(10):
                print('nop')
        sock = ReadSocket('HTTP/1.1 200 OK\r\nContent-Type: text/plain; charset=UTF-8\r\nContent-Length: 18\n\r\n0123456789\ngarbage')
        conn = FakeHTTPConnection(sock)
        conn.putrequest('GET', 'http://localhost/fictious')
        conn.endheaders()
        resp = conn.getresponse()
        self.assertEqual('0123456789\n', resp.read(11))
        conn._range_warning_thresold = 6
        conn.cleanup_pipe()
        self.assertContainsRe(self.get_log(), 'Got a 200 response when asking')

class TestRangeFileMixin(object):
    """Tests for accessing the first range in a RangeFile."""
    alpha = 'abcdefghijklmnopqrstuvwxyz'

    def test_can_read_at_first_access(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the just created file can be read.'
        self.assertEqual(self.alpha, self._file.read())

    def test_seek_read(self):
        if False:
            print('Hello World!')
        'Test seek/read inside the range.'
        f = self._file
        start = self.first_range_start
        self.assertEqual(start, f.tell())
        cur = start
        f.seek(start + 3)
        cur += 3
        self.assertEqual('def', f.read(3))
        cur += len('def')
        f.seek(4, 1)
        cur += 4
        self.assertEqual('klmn', f.read(4))
        cur += len('klmn')
        self.assertEqual('', f.read(0))
        here = f.tell()
        f.seek(0, 1)
        self.assertEqual(here, f.tell())
        self.assertEqual(cur, f.tell())

    def test_read_zero(self):
        if False:
            print('Hello World!')
        f = self._file
        self.assertEqual('', f.read(0))
        f.seek(10, 1)
        self.assertEqual('', f.read(0))

    def test_seek_at_range_end(self):
        if False:
            print('Hello World!')
        f = self._file
        f.seek(26, 1)

    def test_read_at_range_end(self):
        if False:
            for i in range(10):
                print('nop')
        'Test read behaviour at range end.'
        f = self._file
        self.assertEqual(self.alpha, f.read())
        self.assertEqual('', f.read(0))
        self.assertRaises(errors.InvalidRange, f.read, 1)

    def test_unbounded_read_after_seek(self):
        if False:
            for i in range(10):
                print('nop')
        f = self._file
        f.seek(24, 1)
        self.assertEqual('yz', f.read())

    def test_seek_backwards(self):
        if False:
            for i in range(10):
                print('nop')
        f = self._file
        start = self.first_range_start
        f.seek(start)
        f.read(12)
        self.assertRaises(errors.InvalidRange, f.seek, start + 5)

    def test_seek_outside_single_range(self):
        if False:
            print('Hello World!')
        f = self._file
        if f._size == -1 or f._boundary is not None:
            raise tests.TestNotApplicable('Needs a fully defined range')
        self.assertRaises(errors.InvalidRange, f.seek, self.first_range_start + 27)

    def test_read_past_end_of_range(self):
        if False:
            print('Hello World!')
        f = self._file
        if f._size == -1:
            raise tests.TestNotApplicable("Can't check an unknown size")
        start = self.first_range_start
        f.seek(start + 20)
        self.assertRaises(errors.InvalidRange, f.read, 10)

    def test_seek_from_end(self):
        if False:
            i = 10
            return i + 15
        "Test seeking from the end of the file.\n\n       The semantic is unclear in case of multiple ranges. Seeking from end\n       exists only for the http transports, cannot be used if the file size is\n       unknown and is not used in bzrlib itself. This test must be (and is)\n       overridden by daughter classes.\n\n       Reading from end makes sense only when a range has been requested from\n       the end of the file (see HttpTransportBase._get() when using the\n       'tail_amount' parameter). The HTTP response can only be a whole file or\n       a single range.\n       "
        f = self._file
        f.seek(-2, 2)
        self.assertEqual('yz', f.read())

class TestRangeFileSizeUnknown(tests.TestCase, TestRangeFileMixin):
    """Test a RangeFile for a whole file whose size is not known."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestRangeFileSizeUnknown, self).setUp()
        self._file = response.RangeFile('Whole_file_size_known', StringIO(self.alpha))
        self.first_range_start = 0

    def test_seek_from_end(self):
        if False:
            print('Hello World!')
        "See TestRangeFileMixin.test_seek_from_end.\n\n        The end of the file can't be determined since the size is unknown.\n        "
        self.assertRaises(errors.InvalidRange, self._file.seek, -1, 2)

    def test_read_at_range_end(self):
        if False:
            i = 10
            return i + 15
        'Test read behaviour at range end.'
        f = self._file
        self.assertEqual(self.alpha, f.read())
        self.assertEqual('', f.read(0))
        self.assertEqual('', f.read(1))

class TestRangeFileSizeKnown(tests.TestCase, TestRangeFileMixin):
    """Test a RangeFile for a whole file whose size is known."""

    def setUp(self):
        if False:
            print('Hello World!')
        super(TestRangeFileSizeKnown, self).setUp()
        self._file = response.RangeFile('Whole_file_size_known', StringIO(self.alpha))
        self._file.set_range(0, len(self.alpha))
        self.first_range_start = 0

class TestRangeFileSingleRange(tests.TestCase, TestRangeFileMixin):
    """Test a RangeFile for a single range."""

    def setUp(self):
        if False:
            return 10
        super(TestRangeFileSingleRange, self).setUp()
        self._file = response.RangeFile('Single_range_file', StringIO(self.alpha))
        self.first_range_start = 15
        self._file.set_range(self.first_range_start, len(self.alpha))

    def test_read_before_range(self):
        if False:
            while True:
                i = 10
        f = self._file
        f._pos = 0
        self.assertRaises(errors.InvalidRange, f.read, 2)

class TestRangeFileMultipleRanges(tests.TestCase, TestRangeFileMixin):
    """Test a RangeFile for multiple ranges.

    The RangeFile used for the tests contains three ranges:

    - at offset 25: alpha
    - at offset 100: alpha
    - at offset 126: alpha.upper()

    The two last ranges are contiguous. This only rarely occurs (should not in
    fact) in real uses but may lead to hard to track bugs.
    """
    boundary = 'separation'

    def setUp(self):
        if False:
            return 10
        super(TestRangeFileMultipleRanges, self).setUp()
        boundary = self.boundary
        content = ''
        self.first_range_start = 25
        file_size = 200
        for (start, part) in [(self.first_range_start, self.alpha), (100, self.alpha), (126, self.alpha.upper())]:
            content += self._multipart_byterange(part, start, boundary, file_size)
        content += self._boundary_line()
        self._file = response.RangeFile('Multiple_ranges_file', StringIO(content))
        self.set_file_boundary()

    def _boundary_line(self):
        if False:
            print('Hello World!')
        'Helper to build the formatted boundary line.'
        return '--' + self.boundary + '\r\n'

    def set_file_boundary(self):
        if False:
            while True:
                i = 10
        self._file.set_boundary(self.boundary)

    def _multipart_byterange(self, data, offset, boundary, file_size='*'):
        if False:
            for i in range(10):
                print('nop')
        "Encode a part of a file as a multipart/byterange MIME type.\n\n        When a range request is issued, the HTTP response body can be\n        decomposed in parts, each one representing a range (start, size) in a\n        file.\n\n        :param data: The payload.\n        :param offset: where data starts in the file\n        :param boundary: used to separate the parts\n        :param file_size: the size of the file containing the range (default to\n            '*' meaning unknown)\n\n        :return: a string containing the data encoded as it will appear in the\n            HTTP response body.\n        "
        bline = self._boundary_line()
        range = bline
        range += 'Content-Range: bytes %d-%d/%d\r\n' % (offset, offset + len(data) - 1, file_size)
        range += '\r\n'
        range += data
        return range

    def test_read_all_ranges(self):
        if False:
            return 10
        f = self._file
        self.assertEqual(self.alpha, f.read())
        f.seek(100)
        self.assertEqual(self.alpha, f.read())
        self.assertEqual(126, f.tell())
        f.seek(126)
        self.assertEqual('A', f.read(1))
        f.seek(10, 1)
        self.assertEqual('LMN', f.read(3))

    def test_seek_from_end(self):
        if False:
            i = 10
            return i + 15
        'See TestRangeFileMixin.test_seek_from_end.'
        f = self._file
        f.seek(-2, 2)
        self.assertEqual('yz', f.read())
        self.assertRaises(errors.InvalidRange, f.seek, -2, 2)

    def test_seek_into_void(self):
        if False:
            i = 10
            return i + 15
        f = self._file
        start = self.first_range_start
        f.seek(start)
        f.seek(start + 40)
        f.seek(100)
        f.seek(125)

    def test_seek_across_ranges(self):
        if False:
            while True:
                i = 10
        f = self._file
        f.seek(126)
        self.assertEqual('AB', f.read(2))

    def test_checked_read_dont_overflow_buffers(self):
        if False:
            print('Hello World!')
        f = self._file
        f._discarded_buf_size = 8
        f.seek(126)
        self.assertEqual('AB', f.read(2))

    def test_seek_twice_between_ranges(self):
        if False:
            i = 10
            return i + 15
        f = self._file
        start = self.first_range_start
        f.seek(start + 40)
        self.assertRaises(errors.InvalidRange, f.seek, start + 41)

    def test_seek_at_range_end(self):
        if False:
            return 10
        'Test seek behavior at range end.'
        f = self._file
        f.seek(25 + 25)
        f.seek(100 + 25)
        f.seek(126 + 25)

    def test_read_at_range_end(self):
        if False:
            print('Hello World!')
        f = self._file
        self.assertEqual(self.alpha, f.read())
        self.assertEqual(self.alpha, f.read())
        self.assertEqual(self.alpha.upper(), f.read())
        self.assertRaises(errors.InvalidHttpResponse, f.read, 1)

class TestRangeFileMultipleRangesQuotedBoundaries(TestRangeFileMultipleRanges):
    """Perform the same tests as TestRangeFileMultipleRanges, but uses
    an angle-bracket quoted boundary string like IIS 6.0 and 7.0
    (but not IIS 5, which breaks the RFC in a different way
    by using square brackets, not angle brackets)

    This reveals a bug caused by

    - The bad implementation of RFC 822 unquoting in Python (angles are not
      quotes), coupled with

    - The bad implementation of RFC 2046 in IIS (angles are not permitted chars
      in boundary lines).

    """
    _boundary_trimmed = 'q1w2e3r4t5y6u7i8o9p0zaxscdvfbgnhmjklkl'
    boundary = '<' + _boundary_trimmed + '>'

    def set_file_boundary(self):
        if False:
            for i in range(10):
                print('nop')
        self._file.set_boundary(self._boundary_trimmed)

class TestRangeFileVarious(tests.TestCase):
    """Tests RangeFile aspects not covered elsewhere."""

    def test_seek_whence(self):
        if False:
            i = 10
            return i + 15
        'Test the seek whence parameter values.'
        f = response.RangeFile('foo', StringIO('abc'))
        f.set_range(0, 3)
        f.seek(0)
        f.seek(1, 1)
        f.seek(-1, 2)
        self.assertRaises(ValueError, f.seek, 0, 14)

    def test_range_syntax(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the Content-Range scanning.'
        f = response.RangeFile('foo', StringIO())

        def ok(expected, header_value):
            if False:
                for i in range(10):
                    print('nop')
            f.set_range_from_header(header_value)
            self.assertEqual(expected, (f.tell(), f._size))
        ok((1, 10), 'bytes 1-10/11')
        ok((1, 10), 'bytes 1-10/*')
        ok((12, 2), '\tbytes 12-13/*')
        ok((28, 1), '  bytes 28-28/*')
        ok((2123, 2120), 'bytes  2123-4242/12310')
        ok((1, 10), 'bytes 1-10/ttt')

        def nok(header_value):
            if False:
                print('Hello World!')
            self.assertRaises(errors.InvalidHttpRange, f.set_range_from_header, header_value)
        nok('bytes 10-2/3')
        nok('chars 1-2/3')
        nok('bytes xx-yyy/zzz')
        nok('bytes xx-12/zzz')
        nok('bytes 11-yy/zzz')
        nok('bytes10-2/3')
_full_text_response = (200, 'HTTP/1.1 200 OK\r\nDate: Tue, 11 Jul 2006 04:32:56 GMT\r\nServer: Apache/2.0.54 (Fedora)\r\nLast-Modified: Sun, 23 Apr 2006 19:35:20 GMT\r\nETag: "56691-23-38e9ae00"\r\nAccept-Ranges: bytes\r\nContent-Length: 35\r\nConnection: close\r\nContent-Type: text/plain; charset=UTF-8\r\n\r\n', 'Bazaar-NG meta directory, format 1\n')
_single_range_response = (206, 'HTTP/1.1 206 Partial Content\r\nDate: Tue, 11 Jul 2006 04:45:22 GMT\r\nServer: Apache/2.0.54 (Fedora)\r\nLast-Modified: Thu, 06 Jul 2006 20:22:05 GMT\r\nETag: "238a3c-16ec2-805c5540"\r\nAccept-Ranges: bytes\r\nContent-Length: 100\r\nContent-Range: bytes 100-199/93890\r\nConnection: close\r\nContent-Type: text/plain; charset=UTF-8\r\n\r\n', 'mbp@sourcefrog.net-20050309040815-13242001617e4a06\nmbp@sourcefrog.net-20050309040929-eee0eb3e6d1e762')
_single_range_no_content_type = (206, 'HTTP/1.1 206 Partial Content\r\nDate: Tue, 11 Jul 2006 04:45:22 GMT\r\nServer: Apache/2.0.54 (Fedora)\r\nLast-Modified: Thu, 06 Jul 2006 20:22:05 GMT\r\nETag: "238a3c-16ec2-805c5540"\r\nAccept-Ranges: bytes\r\nContent-Length: 100\r\nContent-Range: bytes 100-199/93890\r\nConnection: close\r\n\r\n', 'mbp@sourcefrog.net-20050309040815-13242001617e4a06\nmbp@sourcefrog.net-20050309040929-eee0eb3e6d1e762')
_multipart_range_response = (206, 'HTTP/1.1 206 Partial Content\r\nDate: Tue, 11 Jul 2006 04:49:48 GMT\r\nServer: Apache/2.0.54 (Fedora)\r\nLast-Modified: Thu, 06 Jul 2006 20:22:05 GMT\r\nETag: "238a3c-16ec2-805c5540"\r\nAccept-Ranges: bytes\r\nContent-Length: 1534\r\nConnection: close\r\nContent-Type: multipart/byteranges; boundary=418470f848b63279b\r\n\r\n\r', '--418470f848b63279b\r\nContent-type: text/plain; charset=UTF-8\r\nContent-range: bytes 0-254/93890\r\n\r\nmbp@sourcefrog.net-20050309040815-13242001617e4a06\nmbp@sourcefrog.net-20050309040929-eee0eb3e6d1e7627\nmbp@sourcefrog.net-20050309040957-6cad07f466bb0bb8\nmbp@sourcefrog.net-20050309041501-c840e09071de3b67\nmbp@sourcefrog.net-20050309044615-c24a3250be83220a\n\r\n--418470f848b63279b\r\nContent-type: text/plain; charset=UTF-8\r\nContent-range: bytes 1000-2049/93890\r\n\r\n40-fd4ec249b6b139ab\nmbp@sourcefrog.net-20050311063625-07858525021f270b\nmbp@sourcefrog.net-20050311231934-aa3776aff5200bb9\nmbp@sourcefrog.net-20050311231953-73aeb3a131c3699a\nmbp@sourcefrog.net-20050311232353-f5e33da490872c6a\nmbp@sourcefrog.net-20050312071639-0a8f59a34a024ff0\nmbp@sourcefrog.net-20050312073432-b2c16a55e0d6e9fb\nmbp@sourcefrog.net-20050312073831-a47c3335ece1920f\nmbp@sourcefrog.net-20050312085412-13373aa129ccbad3\nmbp@sourcefrog.net-20050313052251-2bf004cb96b39933\nmbp@sourcefrog.net-20050313052856-3edd84094687cb11\nmbp@sourcefrog.net-20050313053233-e30a4f28aef48f9d\nmbp@sourcefrog.net-20050313053853-7c64085594ff3072\nmbp@sourcefrog.net-20050313054757-a86c3f5871069e22\nmbp@sourcefrog.net-20050313061422-418f1f73b94879b9\nmbp@sourcefrog.net-20050313120651-497bd231b19df600\nmbp@sourcefrog.net-20050314024931-eae0170ef25a5d1a\nmbp@sourcefrog.net-20050314025438-d52099f915fe65fc\nmbp@sourcefrog.net-20050314025539-637a636692c055cf\nmbp@sourcefrog.net-20050314025737-55eb441f430ab4ba\nmbp@sourcefrog.net-20050314025901-d74aa93bb7ee8f62\nmbp@source\r\n--418470f848b63279b--\r\n')
_multipart_squid_range_response = (206, 'HTTP/1.0 206 Partial Content\r\nDate: Thu, 31 Aug 2006 21:16:22 GMT\r\nServer: Apache/2.2.2 (Unix) DAV/2\r\nLast-Modified: Thu, 31 Aug 2006 17:57:06 GMT\r\nAccept-Ranges: bytes\r\nContent-Type: multipart/byteranges; boundary="squid/2.5.STABLE12:C99323425AD4FE26F726261FA6C24196"\r\nContent-Length: 598\r\nX-Cache: MISS from localhost.localdomain\r\nX-Cache-Lookup: HIT from localhost.localdomain:3128\r\nProxy-Connection: keep-alive\r\n\r\n', '\r\n--squid/2.5.STABLE12:C99323425AD4FE26F726261FA6C24196\r\nContent-Type: text/plain\r\nContent-Range: bytes 0-99/18672\r\n\r\n# bzr knit index 8\n\nscott@netsplit.com-20050708230047-47c7868f276b939f fulltext 0 863  :\nscott@netsp\r\n--squid/2.5.STABLE12:C99323425AD4FE26F726261FA6C24196\r\nContent-Type: text/plain\r\nContent-Range: bytes 300-499/18672\r\n\r\ncom-20050708231537-2b124b835395399a :\nscott@netsplit.com-20050820234126-551311dbb7435b51 line-delta 1803 479 .scott@netsplit.com-20050820232911-dc4322a084eadf7e :\nscott@netsplit.com-20050821213706-c86\r\n--squid/2.5.STABLE12:C99323425AD4FE26F726261FA6C24196--\r\n')
_full_text_response_no_content_type = (200, 'HTTP/1.1 200 OK\r\nDate: Tue, 11 Jul 2006 04:32:56 GMT\r\nServer: Apache/2.0.54 (Fedora)\r\nLast-Modified: Sun, 23 Apr 2006 19:35:20 GMT\r\nETag: "56691-23-38e9ae00"\r\nAccept-Ranges: bytes\r\nContent-Length: 35\r\nConnection: close\r\n\r\n', 'Bazaar-NG meta directory, format 1\n')
_full_text_response_no_content_length = (200, 'HTTP/1.1 200 OK\r\nDate: Tue, 11 Jul 2006 04:32:56 GMT\r\nServer: Apache/2.0.54 (Fedora)\r\nLast-Modified: Sun, 23 Apr 2006 19:35:20 GMT\r\nETag: "56691-23-38e9ae00"\r\nAccept-Ranges: bytes\r\nConnection: close\r\nContent-Type: text/plain; charset=UTF-8\r\n\r\n', 'Bazaar-NG meta directory, format 1\n')
_single_range_no_content_range = (206, 'HTTP/1.1 206 Partial Content\r\nDate: Tue, 11 Jul 2006 04:45:22 GMT\r\nServer: Apache/2.0.54 (Fedora)\r\nLast-Modified: Thu, 06 Jul 2006 20:22:05 GMT\r\nETag: "238a3c-16ec2-805c5540"\r\nAccept-Ranges: bytes\r\nContent-Length: 100\r\nConnection: close\r\n\r\n', 'mbp@sourcefrog.net-20050309040815-13242001617e4a06\nmbp@sourcefrog.net-20050309040929-eee0eb3e6d1e762')
_single_range_response_truncated = (206, 'HTTP/1.1 206 Partial Content\r\nDate: Tue, 11 Jul 2006 04:45:22 GMT\r\nServer: Apache/2.0.54 (Fedora)\r\nLast-Modified: Thu, 06 Jul 2006 20:22:05 GMT\r\nETag: "238a3c-16ec2-805c5540"\r\nAccept-Ranges: bytes\r\nContent-Length: 100\r\nContent-Range: bytes 100-199/93890\r\nConnection: close\r\nContent-Type: text/plain; charset=UTF-8\r\n\r\n', 'mbp@sourcefrog.net-20050309040815-13242001617e4a06')
_invalid_response = (444, 'HTTP/1.1 444 Bad Response\r\nDate: Tue, 11 Jul 2006 04:32:56 GMT\r\nConnection: close\r\nContent-Type: text/html; charset=iso-8859-1\r\n\r\n', '<!DOCTYPE HTML PUBLIC "-//IETF//DTD HTML 2.0//EN">\n<html><head>\n<title>404 Not Found</title>\n</head><body>\n<h1>Not Found</h1>\n<p>I don\'t know what I\'m doing</p>\n<hr>\n</body></html>\n')
_multipart_no_content_range = (206, 'HTTP/1.0 206 Partial Content\r\nContent-Type: multipart/byteranges; boundary=THIS_SEPARATES\r\nContent-Length: 598\r\n\r\n', '\r\n--THIS_SEPARATES\r\nContent-Type: text/plain\r\n\r\n# bzr knit index 8\n--THIS_SEPARATES\r\n')
_multipart_no_boundary = (206, 'HTTP/1.0 206 Partial Content\r\nContent-Type: multipart/byteranges; boundary=THIS_SEPARATES\r\nContent-Length: 598\r\n\r\n', '\r\n--THIS_SEPARATES\r\nContent-Type: text/plain\r\nContent-Range: bytes 0-18/18672\r\n\r\n# bzr knit index 8\n\nThe range ended at the line above, this text is garbage instead of a boundary\nline\n')

class TestHandleResponse(tests.TestCase):

    def _build_HTTPMessage(self, raw_headers):
        if False:
            print('Hello World!')
        status_and_headers = StringIO(raw_headers)
        status_and_headers.readline()
        msg = httplib.HTTPMessage(status_and_headers)
        return msg

    def get_response(self, a_response):
        if False:
            print('Hello World!')
        'Process a supplied response, and return the result.'
        (code, raw_headers, body) = a_response
        msg = self._build_HTTPMessage(raw_headers)
        return response.handle_response('http://foo', code, msg, StringIO(a_response[2]))

    def test_full_text(self):
        if False:
            return 10
        out = self.get_response(_full_text_response)
        self.assertEqual(_full_text_response[2], out.read())

    def test_single_range(self):
        if False:
            while True:
                i = 10
        out = self.get_response(_single_range_response)
        out.seek(100)
        self.assertEqual(_single_range_response[2], out.read(100))

    def test_single_range_no_content(self):
        if False:
            while True:
                i = 10
        out = self.get_response(_single_range_no_content_type)
        out.seek(100)
        self.assertEqual(_single_range_no_content_type[2], out.read(100))

    def test_single_range_truncated(self):
        if False:
            for i in range(10):
                print('nop')
        out = self.get_response(_single_range_response_truncated)
        self.assertRaises(errors.ShortReadvError, out.seek, out.tell() + 51)

    def test_multi_range(self):
        if False:
            return 10
        out = self.get_response(_multipart_range_response)
        out.seek(0)
        out.read(255)
        out.seek(1000)
        out.read(1050)

    def test_multi_squid_range(self):
        if False:
            return 10
        out = self.get_response(_multipart_squid_range_response)
        out.seek(0)
        out.read(100)
        out.seek(300)
        out.read(200)

    def test_invalid_response(self):
        if False:
            while True:
                i = 10
        self.assertRaises(errors.InvalidHttpResponse, self.get_response, _invalid_response)

    def test_full_text_no_content_type(self):
        if False:
            while True:
                i = 10
        (code, raw_headers, body) = _full_text_response_no_content_type
        msg = self._build_HTTPMessage(raw_headers)
        out = response.handle_response('http://foo', code, msg, StringIO(body))
        self.assertEqual(body, out.read())

    def test_full_text_no_content_length(self):
        if False:
            for i in range(10):
                print('nop')
        (code, raw_headers, body) = _full_text_response_no_content_length
        msg = self._build_HTTPMessage(raw_headers)
        out = response.handle_response('http://foo', code, msg, StringIO(body))
        self.assertEqual(body, out.read())

    def test_missing_content_range(self):
        if False:
            while True:
                i = 10
        (code, raw_headers, body) = _single_range_no_content_range
        msg = self._build_HTTPMessage(raw_headers)
        self.assertRaises(errors.InvalidHttpResponse, response.handle_response, 'http://bogus', code, msg, StringIO(body))

    def test_multipart_no_content_range(self):
        if False:
            for i in range(10):
                print('nop')
        (code, raw_headers, body) = _multipart_no_content_range
        msg = self._build_HTTPMessage(raw_headers)
        self.assertRaises(errors.InvalidHttpResponse, response.handle_response, 'http://bogus', code, msg, StringIO(body))

    def test_multipart_no_boundary(self):
        if False:
            for i in range(10):
                print('nop')
        out = self.get_response(_multipart_no_boundary)
        out.read()
        self.assertRaises(errors.InvalidHttpResponse, out.seek, 1, 1)

class TestRangeFileSizeReadLimited(tests.TestCase):
    """Test RangeFile _max_read_size functionality which limits the size of
    read blocks to prevent MemoryError messages in socket.recv.
    """

    def setUp(self):
        if False:
            return 10
        super(TestRangeFileSizeReadLimited, self).setUp()
        chunk_size = response.RangeFile._max_read_size
        test_pattern = '0123456789ABCDEF'
        self.test_data = test_pattern * (3 * chunk_size / len(test_pattern))
        self.test_data_len = len(self.test_data)

    def test_max_read_size(self):
        if False:
            while True:
                i = 10
        'Read data in blocks and verify that the reads are not larger than\n           the maximum read size.\n        '
        mock_read_file = FakeReadFile(self.test_data)
        range_file = response.RangeFile('test_max_read_size', mock_read_file)
        response_data = range_file.read(self.test_data_len)
        self.assertTrue(mock_read_file.get_max_read_size() > 0)
        self.assertEqual(mock_read_file.get_max_read_size(), response.RangeFile._max_read_size)
        self.assertEqual(mock_read_file.get_read_count(), 3)
        if response_data != self.test_data:
            message = 'Data not equal.  Expected %d bytes, received %d.'
            self.fail(message % (len(response_data), self.test_data_len))