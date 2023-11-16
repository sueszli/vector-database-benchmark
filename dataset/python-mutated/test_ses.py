from __future__ import annotations
import boto3
import pytest
from moto import mock_ses
from airflow.providers.amazon.aws.hooks.ses import SesHook
boto3.setup_default_session()

@mock_ses
def test_get_conn():
    if False:
        return 10
    hook = SesHook(aws_conn_id='aws_default')
    assert hook.get_conn() is not None

@mock_ses
@pytest.mark.parametrize('to', ['to@domain.com', ['to1@domain.com', 'to2@domain.com'], 'to1@domain.com,to2@domain.com'])
@pytest.mark.parametrize('cc', ['cc@domain.com', ['cc1@domain.com', 'cc2@domain.com'], 'cc1@domain.com,cc2@domain.com'])
@pytest.mark.parametrize('bcc', ['bcc@domain.com', ['bcc1@domain.com', 'bcc2@domain.com'], 'bcc1@domain.com,bcc2@domain.com'])
def test_send_email(to, cc, bcc):
    if False:
        for i in range(10):
            print('nop')
    hook = SesHook()
    ses_client = hook.get_conn()
    mail_from = 'test_from@domain.com'
    ses_client.verify_email_identity(EmailAddress=mail_from)
    response = hook.send_email(mail_from=mail_from, to=to, subject='subject', html_content='<html>Test</html>', cc=cc, bcc=bcc, reply_to='reply_to@domain.com', return_path='return_path@domain.com')
    assert response is not None
    assert isinstance(response, dict)
    assert 'MessageId' in response