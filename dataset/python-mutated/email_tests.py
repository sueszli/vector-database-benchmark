"""Unit tests for email service in Superset"""
import logging
import ssl
import tempfile
import unittest
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from unittest import mock
from superset import app
from superset.utils import core as utils
from tests.integration_tests.base_tests import SupersetTestCase
from .utils import read_fixture
send_email_test = mock.Mock()
logger = logging.getLogger(__name__)

class TestEmailSmtp(SupersetTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        app.config['SMTP_SSL'] = False

    @mock.patch('superset.utils.core.send_mime_email')
    def test_send_smtp(self, mock_send_mime):
        if False:
            i = 10
            return i + 15
        attachment = tempfile.NamedTemporaryFile()
        attachment.write(b'attachment')
        attachment.seek(0)
        utils.send_email_smtp('to', 'subject', 'content', app.config, files=[attachment.name])
        assert mock_send_mime.called
        call_args = mock_send_mime.call_args[0]
        logger.debug(call_args)
        assert call_args[0] == app.config['SMTP_MAIL_FROM']
        assert call_args[1] == ['to']
        msg = call_args[2]
        assert msg['Subject'] == 'subject'
        assert msg['From'] == app.config['SMTP_MAIL_FROM']
        assert len(msg.get_payload()) == 2
        mimeapp = MIMEApplication('attachment')
        assert msg.get_payload()[-1].get_payload() == mimeapp.get_payload()

    @mock.patch('superset.utils.core.send_mime_email')
    def test_send_smtp_with_email_mutator(self, mock_send_mime):
        if False:
            while True:
                i = 10
        attachment = tempfile.NamedTemporaryFile()
        attachment.write(b'attachment')
        attachment.seek(0)
        base_email_mutator = app.config['EMAIL_HEADER_MUTATOR']

        def mutator(msg, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            msg['foo'] = 'bar'
            return msg
        app.config['EMAIL_HEADER_MUTATOR'] = mutator
        utils.send_email_smtp('to', 'subject', 'content', app.config, files=[attachment.name])
        assert mock_send_mime.called
        call_args = mock_send_mime.call_args[0]
        logger.debug(call_args)
        assert call_args[0] == app.config['SMTP_MAIL_FROM']
        assert call_args[1] == ['to']
        msg = call_args[2]
        assert msg['Subject'] == 'subject'
        assert msg['From'] == app.config['SMTP_MAIL_FROM']
        assert msg['foo'] == 'bar'
        assert len(msg.get_payload()) == 2
        mimeapp = MIMEApplication('attachment')
        assert msg.get_payload()[-1].get_payload() == mimeapp.get_payload()
        app.config['EMAIL_HEADER_MUTATOR'] = base_email_mutator

    @mock.patch('superset.utils.core.send_mime_email')
    def test_send_smtp_data(self, mock_send_mime):
        if False:
            while True:
                i = 10
        utils.send_email_smtp('to', 'subject', 'content', app.config, data={'1.txt': b'data'})
        assert mock_send_mime.called
        call_args = mock_send_mime.call_args[0]
        logger.debug(call_args)
        assert call_args[0] == app.config['SMTP_MAIL_FROM']
        assert call_args[1] == ['to']
        msg = call_args[2]
        assert msg['Subject'] == 'subject'
        assert msg['From'] == app.config['SMTP_MAIL_FROM']
        assert len(msg.get_payload()) == 2
        mimeapp = MIMEApplication('data')
        assert msg.get_payload()[-1].get_payload() == mimeapp.get_payload()

    @mock.patch('superset.utils.core.send_mime_email')
    def test_send_smtp_inline_images(self, mock_send_mime):
        if False:
            i = 10
            return i + 15
        image = read_fixture('sample.png')
        utils.send_email_smtp('to', 'subject', 'content', app.config, images=dict(blah=image))
        assert mock_send_mime.called
        call_args = mock_send_mime.call_args[0]
        logger.debug(call_args)
        assert call_args[0] == app.config['SMTP_MAIL_FROM']
        assert call_args[1] == ['to']
        msg = call_args[2]
        assert msg['Subject'] == 'subject'
        assert msg['From'] == app.config['SMTP_MAIL_FROM']
        assert len(msg.get_payload()) == 2
        mimeapp = MIMEImage(image)
        assert msg.get_payload()[-1].get_payload() == mimeapp.get_payload()

    @mock.patch('superset.utils.core.send_mime_email')
    def test_send_bcc_smtp(self, mock_send_mime):
        if False:
            for i in range(10):
                print('nop')
        attachment = tempfile.NamedTemporaryFile()
        attachment.write(b'attachment')
        attachment.seek(0)
        utils.send_email_smtp('to', 'subject', 'content', app.config, files=[attachment.name], cc='cc', bcc='bcc')
        assert mock_send_mime.called
        call_args = mock_send_mime.call_args[0]
        assert call_args[0] == app.config['SMTP_MAIL_FROM']
        assert call_args[1] == ['to', 'cc', 'bcc']
        msg = call_args[2]
        assert msg['Subject'] == 'subject'
        assert msg['From'] == app.config['SMTP_MAIL_FROM']
        assert len(msg.get_payload()) == 2
        mimeapp = MIMEApplication('attachment')
        assert msg.get_payload()[-1].get_payload() == mimeapp.get_payload()

    @mock.patch('smtplib.SMTP_SSL')
    @mock.patch('smtplib.SMTP')
    def test_send_mime(self, mock_smtp, mock_smtp_ssl):
        if False:
            print('Hello World!')
        mock_smtp.return_value = mock.Mock()
        mock_smtp_ssl.return_value = mock.Mock()
        msg = MIMEMultipart()
        utils.send_mime_email('from', 'to', msg, app.config, dryrun=False)
        mock_smtp.assert_called_with(app.config['SMTP_HOST'], app.config['SMTP_PORT'])
        assert mock_smtp.return_value.starttls.called
        mock_smtp.return_value.login.assert_called_with(app.config['SMTP_USER'], app.config['SMTP_PASSWORD'])
        mock_smtp.return_value.sendmail.assert_called_with('from', 'to', msg.as_string())
        assert mock_smtp.return_value.quit.called

    @mock.patch('smtplib.SMTP_SSL')
    @mock.patch('smtplib.SMTP')
    def test_send_mime_ssl(self, mock_smtp, mock_smtp_ssl):
        if False:
            for i in range(10):
                print('nop')
        app.config['SMTP_SSL'] = True
        mock_smtp.return_value = mock.Mock()
        mock_smtp_ssl.return_value = mock.Mock()
        utils.send_mime_email('from', 'to', MIMEMultipart(), app.config, dryrun=False)
        assert not mock_smtp.called
        mock_smtp_ssl.assert_called_with(app.config['SMTP_HOST'], app.config['SMTP_PORT'], context=None)

    @mock.patch('smtplib.SMTP_SSL')
    @mock.patch('smtplib.SMTP')
    def test_send_mime_ssl_server_auth(self, mock_smtp, mock_smtp_ssl):
        if False:
            return 10
        app.config['SMTP_SSL'] = True
        app.config['SMTP_SSL_SERVER_AUTH'] = True
        mock_smtp.return_value = mock.Mock()
        mock_smtp_ssl.return_value = mock.Mock()
        utils.send_mime_email('from', 'to', MIMEMultipart(), app.config, dryrun=False)
        assert not mock_smtp.called
        mock_smtp_ssl.assert_called_with(app.config['SMTP_HOST'], app.config['SMTP_PORT'], context=mock.ANY)
        called_context = mock_smtp_ssl.call_args.kwargs['context']
        self.assertEqual(called_context.verify_mode, ssl.CERT_REQUIRED)

    @mock.patch('smtplib.SMTP')
    def test_send_mime_tls_server_auth(self, mock_smtp):
        if False:
            print('Hello World!')
        app.config['SMTP_STARTTLS'] = True
        app.config['SMTP_SSL_SERVER_AUTH'] = True
        mock_smtp.return_value = mock.Mock()
        mock_smtp.return_value.starttls.return_value = mock.Mock()
        utils.send_mime_email('from', 'to', MIMEMultipart(), app.config, dryrun=False)
        mock_smtp.return_value.starttls.assert_called_with(context=mock.ANY)
        called_context = mock_smtp.return_value.starttls.call_args.kwargs['context']
        self.assertEqual(called_context.verify_mode, ssl.CERT_REQUIRED)

    @mock.patch('smtplib.SMTP_SSL')
    @mock.patch('smtplib.SMTP')
    def test_send_mime_noauth(self, mock_smtp, mock_smtp_ssl):
        if False:
            while True:
                i = 10
        smtp_user = app.config['SMTP_USER']
        smtp_password = app.config['SMTP_PASSWORD']
        app.config['SMTP_USER'] = None
        app.config['SMTP_PASSWORD'] = None
        mock_smtp.return_value = mock.Mock()
        mock_smtp_ssl.return_value = mock.Mock()
        utils.send_mime_email('from', 'to', MIMEMultipart(), app.config, dryrun=False)
        assert not mock_smtp_ssl.called
        mock_smtp.assert_called_with(app.config['SMTP_HOST'], app.config['SMTP_PORT'])
        assert not mock_smtp.login.called
        app.config['SMTP_USER'] = smtp_user
        app.config['SMTP_PASSWORD'] = smtp_password

    @mock.patch('smtplib.SMTP_SSL')
    @mock.patch('smtplib.SMTP')
    def test_send_mime_dryrun(self, mock_smtp, mock_smtp_ssl):
        if False:
            i = 10
            return i + 15
        utils.send_mime_email('from', 'to', MIMEMultipart(), app.config, dryrun=True)
        assert not mock_smtp.called
        assert not mock_smtp_ssl.called
if __name__ == '__main__':
    unittest.main()