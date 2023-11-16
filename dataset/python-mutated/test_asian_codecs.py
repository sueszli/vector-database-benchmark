import unittest
from test.test_email import TestEmailBase
from email.charset import Charset
from email.header import Header, decode_header
from email.message import Message
try:
    str(b'foo', 'euc-jp')
except LookupError:
    raise unittest.SkipTest

class TestEmailAsianCodecs(TestEmailBase):

    def test_japanese_codecs(self):
        if False:
            i = 10
            return i + 15
        eq = self.ndiffAssertEqual
        jcode = 'euc-jp'
        gcode = 'iso-8859-1'
        j = Charset(jcode)
        g = Charset(gcode)
        h = Header('Hello World!')
        jhello = str(b'\xa5\xcf\xa5\xed\xa1\xbc\xa5\xef\xa1\xbc\xa5\xeb\xa5\xc9\xa1\xaa', jcode)
        ghello = str(b'Gr\xfc\xdf Gott!', gcode)
        h.append(jhello, j)
        h.append(ghello, g)
        eq(h.encode(), 'Hello World! =?iso-2022-jp?b?GyRCJU8lbSE8JW8hPCVrJUkhKhsoQg==?=\n =?iso-8859-1?q?Gr=FC=DF_Gott!?=')
        eq(decode_header(h.encode()), [(b'Hello World! ', None), (b'\x1b$B%O%m!<%o!<%k%I!*\x1b(B', 'iso-2022-jp'), (b'Gr\xfc\xdf Gott!', gcode)])
        subject_bytes = b'test-ja \xa4\xd8\xc5\xea\xb9\xc6\xa4\xb5\xa4\xec\xa4\xbf\xa5\xe1\xa1\xbc\xa5\xeb\xa4\xcf\xbb\xca\xb2\xf1\xbc\xd4\xa4\xce\xbe\xb5\xc7\xa7\xa4\xf2\xc2\xd4\xa4\xc3\xa4\xc6\xa4\xa4\xa4\xde\xa4\xb9'
        subject = str(subject_bytes, jcode)
        h = Header(subject, j, header_name='Subject')
        enc = h.encode()
        eq(enc, '=?iso-2022-jp?b?dGVzdC1qYSAbJEIkWEVqOUYkNSRsJD8lYSE8JWskTztKGyhC?=\n =?iso-2022-jp?b?GyRCMnE8VCROPjVHJyRyQlQkQyRGJCQkXiQ5GyhC?=')
        eq(str(h).encode(jcode), subject_bytes)

    def test_payload_encoding_utf8(self):
        if False:
            print('Hello World!')
        jhello = str(b'\xa5\xcf\xa5\xed\xa1\xbc\xa5\xef\xa1\xbc\xa5\xeb\xa5\xc9\xa1\xaa', 'euc-jp')
        msg = Message()
        msg.set_payload(jhello, 'utf-8')
        ustr = msg.get_payload(decode=True).decode(msg.get_content_charset())
        self.assertEqual(jhello, ustr)

    def test_payload_encoding(self):
        if False:
            return 10
        jcode = 'euc-jp'
        jhello = str(b'\xa5\xcf\xa5\xed\xa1\xbc\xa5\xef\xa1\xbc\xa5\xeb\xa5\xc9\xa1\xaa', jcode)
        msg = Message()
        msg.set_payload(jhello, jcode)
        ustr = msg.get_payload(decode=True).decode(msg.get_content_charset())
        self.assertEqual(jhello, ustr)
if __name__ == '__main__':
    unittest.main()