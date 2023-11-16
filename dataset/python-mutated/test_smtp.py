from __future__ import annotations
import json
from unittest.mock import patch
from airflow.models import Connection
from airflow.providers.smtp.operators.smtp import EmailOperator
smtplib_string = 'airflow.providers.smtp.hooks.smtp.smtplib'

class TestEmailOperator:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.default_op_kwargs = dict(to='to', subject='subject', html_content='content')

    @patch('airflow.providers.smtp.hooks.smtp.SmtpHook.get_connection')
    @patch(smtplib_string)
    def test_loading_sender_email_from_connection(self, mock_smtplib, mock_hook_conn):
        if False:
            while True:
                i = 10
        'Check if the EmailOperator is able to load the sender email from the smtp connection.'
        custom_retry_limit = 10
        custom_timeout = 60
        sender_email = 'sender_email'
        mock_hook_conn.return_value = Connection(conn_id='mock_conn', conn_type='smtp', host='smtp_server_address', login='smtp_user', password='smtp_password', port=465, extra=json.dumps(dict(from_email=sender_email, timeout=custom_timeout, retry_limit=custom_retry_limit)))
        smtp_client_mock = mock_smtplib.SMTP_SSL()
        op = EmailOperator(task_id='test_email', **self.default_op_kwargs)
        op.execute({})
        call_args = smtp_client_mock.sendmail.call_args.kwargs
        assert call_args['from_addr'] == sender_email

    def test_assert_templated_fields(self):
        if False:
            return 10
        'Test expected templated fields.'
        operator = EmailOperator(task_id='test_assert_templated_fields', **self.default_op_kwargs)
        template_fields = ('to', 'from_email', 'subject', 'html_content', 'files', 'cc', 'bcc')
        assert operator.template_fields == template_fields