import sys
from email.Header import decode_header
from bzrlib import __version__ as _bzrlib_version
from bzrlib.email_message import EmailMessage
from bzrlib.errors import BzrBadParameterNotUnicode
from bzrlib.smtp_connection import SMTPConnection
from bzrlib import tests
EMPTY_MESSAGE = 'From: from@from.com\nSubject: subject\nTo: to@to.com\nUser-Agent: Bazaar (%s)\n\n' % _bzrlib_version
_SIMPLE_MESSAGE = 'MIME-Version: 1.0\nContent-Type: text/plain; charset="%%s"\nContent-Transfer-Encoding: %%s\nFrom: from@from.com\nSubject: subject\nTo: to@to.com\nUser-Agent: Bazaar (%s)\n\n%%s' % _bzrlib_version
SIMPLE_MESSAGE_ASCII = _SIMPLE_MESSAGE % ('us-ascii', '7bit', 'body')
SIMPLE_MESSAGE_UTF8 = _SIMPLE_MESSAGE % ('utf-8', 'base64', 'YsOzZHk=\n')
SIMPLE_MESSAGE_8BIT = _SIMPLE_MESSAGE % ('8-bit', 'base64', 'YvRkeQ==\n')
BOUNDARY = '=====123456=='
_MULTIPART_HEAD = 'Content-Type: multipart/mixed; boundary="%(boundary)s"\nMIME-Version: 1.0\nFrom: from@from.com\nSubject: subject\nTo: to@to.com\nUser-Agent: Bazaar (%(version)s)\n\n--%(boundary)s\nMIME-Version: 1.0\nContent-Type: text/plain; charset="us-ascii"\nContent-Transfer-Encoding: 7bit\nContent-Disposition: inline\n\nbody\n' % {'version': _bzrlib_version, 'boundary': BOUNDARY}

def final_newline_or_not(msg):
    if False:
        i = 10
        return i + 15
    if sys.version_info >= (2, 7, 6):
        msg += '\n'
    return msg

def simple_multipart_message():
    if False:
        while True:
            i = 10
    msg = _MULTIPART_HEAD + '--%s--' % BOUNDARY
    return final_newline_or_not(msg)

def complex_multipart_message(typ):
    if False:
        print('Hello World!')
    msg = _MULTIPART_HEAD + '--%(boundary)s\nMIME-Version: 1.0\nContent-Type: text/%%s; charset="us-ascii"; name="lines.txt"\nContent-Transfer-Encoding: 7bit\nContent-Disposition: inline\n\na\nb\nc\nd\ne\n\n--%(boundary)s--' % {'boundary': BOUNDARY}
    msg = final_newline_or_not(msg)
    return msg % (typ,)

class TestEmailMessage(tests.TestCase):

    def test_empty_message(self):
        if False:
            for i in range(10):
                print('nop')
        msg = EmailMessage('from@from.com', 'to@to.com', 'subject')
        self.assertEqualDiff(EMPTY_MESSAGE, msg.as_string())

    def test_simple_message(self):
        if False:
            print('Hello World!')
        pairs = {'body': SIMPLE_MESSAGE_ASCII, u'bódy': SIMPLE_MESSAGE_UTF8, 'bÃ³dy': SIMPLE_MESSAGE_UTF8, 'bôdy': SIMPLE_MESSAGE_8BIT}
        for (body, expected) in pairs.items():
            msg = EmailMessage('from@from.com', 'to@to.com', 'subject', body)
            self.assertEqualDiff(expected, msg.as_string())

    def test_multipart_message_simple(self):
        if False:
            i = 10
            return i + 15
        msg = EmailMessage('from@from.com', 'to@to.com', 'subject')
        msg.add_inline_attachment('body')
        self.assertEqualDiff(simple_multipart_message(), msg.as_string(BOUNDARY))

    def test_multipart_message_complex(self):
        if False:
            return 10
        msg = EmailMessage('from@from.com', 'to@to.com', 'subject', 'body')
        msg.add_inline_attachment(u'a\nb\nc\nd\ne\n', 'lines.txt', 'x-subtype')
        self.assertEqualDiff(complex_multipart_message('x-subtype'), msg.as_string(BOUNDARY))

    def test_headers_accept_unicode_and_utf8(self):
        if False:
            for i in range(10):
                print('nop')
        for user in [u'Pepe Pérez <pperez@ejemplo.com>', 'Pepe PÃ©red <pperez@ejemplo.com>']:
            msg = EmailMessage(user, user, user)
            for header in ['From', 'To', 'Subject']:
                value = msg[header]
                str(value).decode('ascii')

    def test_headers_reject_8bit(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(3):
            x = ['"J. Random Developer" <jrandom@example.com>'] * 3
            x[i] = 'Pepe Pérez <pperez@ejemplo.com>'
            self.assertRaises(BzrBadParameterNotUnicode, EmailMessage, *x)

    def test_multiple_destinations(self):
        if False:
            print('Hello World!')
        to_addresses = ['to1@to.com', 'to2@to.com', 'to3@to.com']
        msg = EmailMessage('from@from.com', to_addresses, 'subject')
        self.assertContainsRe(msg.as_string(), 'To: ' + ', '.join(to_addresses))

    def test_retrieving_headers(self):
        if False:
            return 10
        msg = EmailMessage('from@from.com', 'to@to.com', 'subject')
        for (header, value) in [('From', 'from@from.com'), ('To', 'to@to.com'), ('Subject', 'subject')]:
            self.assertEqual(value, msg.get(header))
            self.assertEqual(value, msg[header])
        self.assertEqual(None, msg.get('Does-Not-Exist'))
        self.assertEqual(None, msg['Does-Not-Exist'])
        self.assertEqual('None', msg.get('Does-Not-Exist', 'None'))

    def test_setting_headers(self):
        if False:
            print('Hello World!')
        msg = EmailMessage('from@from.com', 'to@to.com', 'subject')
        msg['To'] = 'to2@to.com'
        msg['Cc'] = 'cc@cc.com'
        self.assertEqual('to2@to.com', msg['To'])
        self.assertEqual('cc@cc.com', msg['Cc'])

    def test_address_to_encoded_header(self):
        if False:
            i = 10
            return i + 15

        def decode(s):
            if False:
                print('Hello World!')
            'Convert a RFC2047-encoded string to a unicode string.'
            return ' '.join([chunk.decode(encoding or 'ascii') for (chunk, encoding) in decode_header(s)])
        address = 'jrandom@example.com'
        encoded = EmailMessage.address_to_encoded_header(address)
        self.assertEqual(address, encoded)
        address = 'J Random Developer <jrandom@example.com>'
        encoded = EmailMessage.address_to_encoded_header(address)
        self.assertEqual(address, encoded)
        address = '"J. Random Developer" <jrandom@example.com>'
        encoded = EmailMessage.address_to_encoded_header(address)
        self.assertEqual(address, encoded)
        address = u'Pepe Pérez <pperez@ejemplo.com>'
        encoded = EmailMessage.address_to_encoded_header(address)
        self.assertTrue('pperez@ejemplo.com' in encoded)
        self.assertEqual(address, decode(encoded))
        address = 'Pepe PÃ©red <pperez@ejemplo.com>'
        encoded = EmailMessage.address_to_encoded_header(address)
        self.assertTrue('pperez@ejemplo.com' in encoded)
        self.assertEqual(address, decode(encoded).encode('utf-8'))
        address = 'Pepe Pérez <pperez@ejemplo.com>'
        self.assertRaises(BzrBadParameterNotUnicode, EmailMessage.address_to_encoded_header, address)

    def test_string_with_encoding(self):
        if False:
            for i in range(10):
                print('nop')
        pairs = {u'Pepe': ('Pepe', 'ascii'), u'Pérez': ('PÃ©rez', 'utf-8'), 'Perez': ('Perez', 'ascii'), 'PÃ©rez': ('PÃ©rez', 'utf-8'), 'Pèrez': ('Pèrez', '8-bit')}
        for (string_, pair) in pairs.items():
            self.assertEqual(pair, EmailMessage.string_with_encoding(string_))

class TestSend(tests.TestCase):

    def setUp(self):
        if False:
            return 10
        super(TestSend, self).setUp()
        self.messages = []

        def send_as_append(_self, msg):
            if False:
                print('Hello World!')
            self.messages.append(msg.as_string(BOUNDARY))
        self.overrideAttr(SMTPConnection, 'send_email', send_as_append)

    def send_email(self, attachment=None, attachment_filename=None, attachment_mime_subtype='plain'):
        if False:
            print('Hello World!')

        class FakeConfig:

            def get(self, option):
                if False:
                    print('Hello World!')
                return None
        EmailMessage.send(FakeConfig(), 'from@from.com', 'to@to.com', 'subject', 'body', attachment=attachment, attachment_filename=attachment_filename, attachment_mime_subtype=attachment_mime_subtype)

    def assertMessage(self, expected):
        if False:
            i = 10
            return i + 15
        self.assertLength(1, self.messages)
        self.assertEqualDiff(expected, self.messages[0])

    def test_send_plain(self):
        if False:
            print('Hello World!')
        self.send_email(u'a\nb\nc\nd\ne\n', 'lines.txt')
        self.assertMessage(complex_multipart_message('plain'))

    def test_send_patch(self):
        if False:
            while True:
                i = 10
        self.send_email(u'a\nb\nc\nd\ne\n', 'lines.txt', 'x-patch')
        self.assertMessage(complex_multipart_message('x-patch'))

    def test_send_simple(self):
        if False:
            i = 10
            return i + 15
        self.send_email()
        self.assertMessage(SIMPLE_MESSAGE_ASCII)