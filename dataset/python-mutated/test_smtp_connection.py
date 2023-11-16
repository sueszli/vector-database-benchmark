from email.Message import Message
import errno
import smtplib
import socket
from bzrlib import config, email_message, errors, smtp_connection, tests, ui

def connection_refuser():
    if False:
        while True:
            i = 10

    def connect(server):
        if False:
            for i in range(10):
                print('nop')
        raise socket.error(errno.ECONNREFUSED, 'Connection Refused')
    smtp = smtplib.SMTP()
    smtp.connect = connect
    return smtp

class StubSMTPFactory(object):
    """A fake SMTP connection to test the connection setup."""

    def __init__(self, fail_on=None, smtp_features=None):
        if False:
            for i in range(10):
                print('nop')
        self._fail_on = fail_on or []
        self._calls = []
        self._smtp_features = smtp_features or []
        self._ehlo_called = False

    def __call__(self):
        if False:
            print('Hello World!')
        return self

    def connect(self, server):
        if False:
            return 10
        self._calls.append(('connect', server))

    def helo(self):
        if False:
            print('Hello World!')
        self._calls.append(('helo',))
        if 'helo' in self._fail_on:
            return (500, 'helo failure')
        else:
            return (200, 'helo success')

    def ehlo(self):
        if False:
            while True:
                i = 10
        self._calls.append(('ehlo',))
        if 'ehlo' in self._fail_on:
            return (500, 'ehlo failure')
        else:
            self._ehlo_called = True
            return (200, 'ehlo success')

    def has_extn(self, extension):
        if False:
            i = 10
            return i + 15
        self._calls.append(('has_extn', extension))
        return self._ehlo_called and extension in self._smtp_features

    def starttls(self):
        if False:
            return 10
        self._calls.append(('starttls',))
        if 'starttls' in self._fail_on:
            return (500, 'starttls failure')
        else:
            self._ehlo_called = True
            return (200, 'starttls success')

class WideOpenSMTPFactory(StubSMTPFactory):
    """A fake smtp server that implements login by accepting anybody."""

    def login(self, user, password):
        if False:
            print('Hello World!')
        self._calls.append(('login', user, password))

class TestSMTPConnection(tests.TestCaseInTempDir):

    def get_connection(self, text, smtp_factory=None):
        if False:
            while True:
                i = 10
        my_config = config.MemoryStack(text)
        return smtp_connection.SMTPConnection(my_config, _smtp_factory=smtp_factory)

    def test_defaults(self):
        if False:
            i = 10
            return i + 15
        conn = self.get_connection('')
        self.assertEqual('localhost', conn._smtp_server)
        self.assertEqual(None, conn._smtp_username)
        self.assertEqual(None, conn._smtp_password)

    def test_smtp_server(self):
        if False:
            while True:
                i = 10
        conn = self.get_connection('smtp_server=host:10')
        self.assertEqual('host:10', conn._smtp_server)

    def test_missing_server(self):
        if False:
            i = 10
            return i + 15
        conn = self.get_connection('', smtp_factory=connection_refuser)
        self.assertRaises(errors.DefaultSMTPConnectionRefused, conn._connect)
        conn = self.get_connection('smtp_server=smtp.example.com', smtp_factory=connection_refuser)
        self.assertRaises(errors.SMTPConnectionRefused, conn._connect)

    def test_smtp_username(self):
        if False:
            for i in range(10):
                print('nop')
        conn = self.get_connection('')
        self.assertIs(None, conn._smtp_username)
        conn = self.get_connection('smtp_username=joebody')
        self.assertEqual(u'joebody', conn._smtp_username)

    def test_smtp_password_from_config(self):
        if False:
            while True:
                i = 10
        conn = self.get_connection('')
        self.assertIs(None, conn._smtp_password)
        conn = self.get_connection('smtp_password=mypass')
        self.assertEqual(u'mypass', conn._smtp_password)

    def test_smtp_password_from_user(self):
        if False:
            print('Hello World!')
        user = 'joe'
        password = 'hispass'
        factory = WideOpenSMTPFactory()
        conn = self.get_connection('[DEFAULT]\nsmtp_username=%s\n' % user, smtp_factory=factory)
        self.assertIs(None, conn._smtp_password)
        ui.ui_factory = ui.CannedInputUIFactory([password])
        conn._connect()
        self.assertEqual(password, conn._smtp_password)

    def test_smtp_password_from_auth_config(self):
        if False:
            for i in range(10):
                print('nop')
        user = 'joe'
        password = 'hispass'
        factory = WideOpenSMTPFactory()
        conn = self.get_connection('[DEFAULT]\nsmtp_username=%s\n' % user, smtp_factory=factory)
        self.assertEqual(user, conn._smtp_username)
        self.assertIs(None, conn._smtp_password)
        conf = config.AuthenticationConfig()
        conf._get_config().update({'smtptest': {'scheme': 'smtp', 'user': user, 'password': password}})
        conf._save()
        conn._connect()
        self.assertEqual(password, conn._smtp_password)

    def test_authenticate_with_byte_strings(self):
        if False:
            while True:
                i = 10
        user = 'joe'
        unicode_pass = u'hìspass'
        utf8_pass = unicode_pass.encode('utf-8')
        factory = WideOpenSMTPFactory()
        conn = self.get_connection('[DEFAULT]\nsmtp_username=%s\nsmtp_password=%s\n' % (user, utf8_pass), smtp_factory=factory)
        self.assertEqual(unicode_pass, conn._smtp_password)
        conn._connect()
        self.assertEqual([('connect', 'localhost'), ('ehlo',), ('has_extn', 'starttls'), ('login', user, utf8_pass)], factory._calls)
        (smtp_username, smtp_password) = factory._calls[-1][1:]
        self.assertIsInstance(smtp_username, str)
        self.assertIsInstance(smtp_password, str)

    def test_create_connection(self):
        if False:
            return 10
        factory = StubSMTPFactory()
        conn = self.get_connection('', smtp_factory=factory)
        conn._create_connection()
        self.assertEqual([('connect', 'localhost'), ('ehlo',), ('has_extn', 'starttls')], factory._calls)

    def test_create_connection_ehlo_fails(self):
        if False:
            print('Hello World!')
        factory = StubSMTPFactory(fail_on=['ehlo'])
        conn = self.get_connection('', smtp_factory=factory)
        conn._create_connection()
        self.assertEqual([('connect', 'localhost'), ('ehlo',), ('helo',), ('has_extn', 'starttls')], factory._calls)

    def test_create_connection_ehlo_helo_fails(self):
        if False:
            for i in range(10):
                print('nop')
        factory = StubSMTPFactory(fail_on=['ehlo', 'helo'])
        conn = self.get_connection('', smtp_factory=factory)
        self.assertRaises(errors.SMTPError, conn._create_connection)
        self.assertEqual([('connect', 'localhost'), ('ehlo',), ('helo',)], factory._calls)

    def test_create_connection_starttls(self):
        if False:
            for i in range(10):
                print('nop')
        factory = StubSMTPFactory(smtp_features=['starttls'])
        conn = self.get_connection('', smtp_factory=factory)
        conn._create_connection()
        self.assertEqual([('connect', 'localhost'), ('ehlo',), ('has_extn', 'starttls'), ('starttls',), ('ehlo',)], factory._calls)

    def test_create_connection_starttls_fails(self):
        if False:
            i = 10
            return i + 15
        factory = StubSMTPFactory(fail_on=['starttls'], smtp_features=['starttls'])
        conn = self.get_connection('', smtp_factory=factory)
        self.assertRaises(errors.SMTPError, conn._create_connection)
        self.assertEqual([('connect', 'localhost'), ('ehlo',), ('has_extn', 'starttls'), ('starttls',)], factory._calls)

    def test_get_message_addresses(self):
        if False:
            return 10
        msg = Message()
        (from_, to) = smtp_connection.SMTPConnection.get_message_addresses(msg)
        self.assertEqual('', from_)
        self.assertEqual([], to)
        msg['From'] = '"J. Random Developer" <jrandom@example.com>'
        msg['To'] = 'John Doe <john@doe.com>, Jane Doe <jane@doe.com>'
        msg['CC'] = u'Pepe Pérez <pperez@ejemplo.com>'
        msg['Bcc'] = 'user@localhost'
        (from_, to) = smtp_connection.SMTPConnection.get_message_addresses(msg)
        self.assertEqual('jrandom@example.com', from_)
        self.assertEqual(sorted(['john@doe.com', 'jane@doe.com', 'pperez@ejemplo.com', 'user@localhost']), sorted(to))
        msg = email_message.EmailMessage('"J. Random Developer" <jrandom@example.com>', ['John Doe <john@doe.com>', 'Jane Doe <jane@doe.com>', u'Pepe Pérez <pperez@ejemplo.com>', 'user@localhost'], 'subject')
        (from_, to) = smtp_connection.SMTPConnection.get_message_addresses(msg)
        self.assertEqual('jrandom@example.com', from_)
        self.assertEqual(sorted(['john@doe.com', 'jane@doe.com', 'pperez@ejemplo.com', 'user@localhost']), sorted(to))

    def test_destination_address_required(self):
        if False:
            print('Hello World!')
        msg = Message()
        msg['From'] = '"J. Random Developer" <jrandom@example.com>'
        self.assertRaises(errors.NoDestinationAddress, smtp_connection.SMTPConnection(config.MemoryStack('')).send_email, msg)
        msg = email_message.EmailMessage('from@from.com', '', 'subject')
        self.assertRaises(errors.NoDestinationAddress, smtp_connection.SMTPConnection(config.MemoryStack('')).send_email, msg)
        msg = email_message.EmailMessage('from@from.com', [], 'subject')
        self.assertRaises(errors.NoDestinationAddress, smtp_connection.SMTPConnection(config.MemoryStack('')).send_email, msg)