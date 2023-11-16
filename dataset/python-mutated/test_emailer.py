from __future__ import annotations
import copy
import os
import tempfile
from unittest import mock
from airflow.providers.sendgrid.utils.emailer import send_email

class TestSendEmailSendGrid:

    def setup_method(self):
        if False:
            return 10
        self.recipients = ['foo@foo.com', 'bar@bar.com']
        self.subject = 'sendgrid-send-email unit test'
        self.html_content = '<b>Foo</b> bar'
        self.carbon_copy = ['foo-cc@foo.com', 'bar-cc@bar.com']
        self.bcc = ['foo-bcc@foo.com', 'bar-bcc@bar.com']
        self.expected_mail_data = {'content': [{'type': 'text/html', 'value': self.html_content}], 'personalizations': [{'cc': [{'email': 'foo-cc@foo.com'}, {'email': 'bar-cc@bar.com'}], 'to': [{'email': 'foo@foo.com'}, {'email': 'bar@bar.com'}], 'bcc': [{'email': 'foo-bcc@foo.com'}, {'email': 'bar-bcc@bar.com'}]}], 'from': {'email': 'foo@bar.com'}, 'subject': 'sendgrid-send-email unit test'}
        self.personalization_custom_args = {'arg1': 'val1', 'arg2': 'val2'}
        self.categories = ['cat1', 'cat2']
        self.expected_mail_data_extras = copy.deepcopy(self.expected_mail_data)
        self.expected_mail_data_extras['personalizations'][0]['custom_args'] = self.personalization_custom_args
        self.expected_mail_data_extras['categories'] = ['cat2', 'cat1']
        self.expected_mail_data_extras['from'] = {'name': 'Foo', 'email': 'foo@bar.com'}
        self.expected_mail_data_sender = copy.deepcopy(self.expected_mail_data)
        self.expected_mail_data_sender['from'] = {'name': 'Foo Bar', 'email': 'foo@foo.bar'}

    @mock.patch.dict('os.environ', SENDGRID_MAIL_FROM='foo@bar.com')
    @mock.patch('airflow.providers.sendgrid.utils.emailer._post_sendgrid_mail')
    def test_send_email_sendgrid_correct_email(self, mock_post):
        if False:
            return 10
        with tempfile.NamedTemporaryFile(mode='wt', suffix='.txt') as f:
            f.write('this is some test data')
            f.flush()
            filename = os.path.basename(f.name)
            expected_mail_data = dict(self.expected_mail_data, attachments=[{'content': 'dGhpcyBpcyBzb21lIHRlc3QgZGF0YQ==', 'content_id': f'<{filename}>', 'disposition': 'attachment', 'filename': filename, 'type': 'text/plain'}])
            send_email(self.recipients, self.subject, self.html_content, cc=self.carbon_copy, bcc=self.bcc, files=[f.name])
            mock_post.assert_called_once_with(expected_mail_data, 'sendgrid_default')

    @mock.patch.dict('os.environ', SENDGRID_MAIL_FROM='foo@bar.com', SENDGRID_MAIL_SENDER='Foo')
    @mock.patch('airflow.providers.sendgrid.utils.emailer._post_sendgrid_mail')
    def test_send_email_sendgrid_correct_email_extras(self, mock_post):
        if False:
            while True:
                i = 10
        send_email(self.recipients, self.subject, self.html_content, cc=self.carbon_copy, bcc=self.bcc, personalization_custom_args=self.personalization_custom_args, categories=self.categories)
        mock_post.assert_called_once_with(self.expected_mail_data_extras, 'sendgrid_default')

    @mock.patch.dict('os.environ', clear=True)
    @mock.patch('airflow.providers.sendgrid.utils.emailer._post_sendgrid_mail')
    def test_send_email_sendgrid_sender(self, mock_post):
        if False:
            while True:
                i = 10
        send_email(self.recipients, self.subject, self.html_content, cc=self.carbon_copy, bcc=self.bcc, from_email='foo@foo.bar', from_name='Foo Bar')
        mock_post.assert_called_once_with(self.expected_mail_data_sender, 'sendgrid_default')