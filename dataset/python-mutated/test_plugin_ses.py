import os
import sys
from unittest import mock
import pytest
import requests
from apprise import Apprise
from apprise import AppriseAttachment
from apprise.plugins.NotifySES import NotifySES
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
if hasattr(sys, 'pypy_version_info'):
    raise pytest.skip(reason='Skipping test cases which stall on PyPy', allow_module_level=True)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
AWS_SES_GOOD_RESPONSE = '\n    <SendRawEmailResponse\n         xmlns="http://ses.amazonaws.com/doc/2010-12-01/">\n      <SendRawEmailResult>\n        <MessageId>\n           010f017d87656ee2-a2ea291f-79ea-\n           44f3-9d25-00d041de3007-000000</MessageId>\n      </SendRawEmailResult>\n      <ResponseMetadata>\n        <RequestId>7abb454e-904b-4e46-a23c-2f4d2fc127a6</RequestId>\n      </ResponseMetadata>\n    </SendRawEmailResponse>\n    '
TEST_ACCESS_KEY_ID = 'AHIAJGNT76XIMXDBIJYA'
TEST_ACCESS_KEY_SECRET = 'bu1dHSdO22pfaaVy/wmNsdljF4C07D3bndi9PQJ9'
TEST_REGION = 'us-east-2'
apprise_url_tests = (('ses://', {'instance': TypeError}), ('ses://:@/', {'instance': TypeError}), ('ses://user@example.com/T1JJ3T3L2', {'instance': TypeError}), ('ses://user@example.com/T1JJ3TD4JD/TIiajkdnlazk7FQ/', {'instance': TypeError}), ('ses://T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcevi7FQ/us-west-2', {'instance': TypeError}), ('ses://user@example.com/T1JJ3TD4JD/TIiajkdnlazk7FQ/user2@example.com', {'instance': TypeError}), ('ses://user@example.com/T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcevi7FQ/us-west-2?reply=invalid-email', {'instance': TypeError, 'requests_response_text': AWS_SES_GOOD_RESPONSE}), ('ses://user@example.com/T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcevi7FQ/us-west-2', {'instance': NotifySES, 'requests_response_text': AWS_SES_GOOD_RESPONSE}), ('ses://user@example.com/T1JJ3TD4JD/TIiajkdnlazk7FQ/us-west-2/user2@example.ca/user3@example.eu', {'instance': NotifySES, 'requests_response_text': AWS_SES_GOOD_RESPONSE, 'privacy_url': 'ses://user@example.com/T...D/****/us-west-2'}), ('ses://user@example.com/T1JJ3T3L2/A1BRTD4JD/TIiajkdnlaevi7FQ/us-east-1?to=user2@example.ca', {'instance': NotifySES, 'requests_response_text': AWS_SES_GOOD_RESPONSE}), ('ses://?from=user@example.com&region=us-west-2&access=T1JJ3T3L2&secret=A1BRTD4JD/TIiajkdnlaevi7FQ&reply=No One <noreply@yahoo.ca>&bcc=user.bcc@example.com,user2.bcc@example.com,invalid-email&cc=user.cc@example.com,user2.cc@example.com,invalid-email&to=user2@example.ca', {'instance': NotifySES, 'requests_response_text': AWS_SES_GOOD_RESPONSE}), ('ses://user@example.com/T1JJ3T3L2/A1BRTD4JD/TIiacevi7FQ/us-west-2/?name=From%20Name&to=user2@example.ca,invalid-email', {'instance': NotifySES, 'requests_response_text': AWS_SES_GOOD_RESPONSE}), ('ses://user@example.com/T1JJ3T3L2/A1BRTD4JD/TIiacevi7FQ/us-west-2/?format=text', {'instance': NotifySES, 'requests_response_text': AWS_SES_GOOD_RESPONSE}), ('ses://user@example.com/T1JJ3T3L2/A1BRTD4JD/TIiacevi7FQ/us-west-2/?to=invalid-email', {'instance': NotifySES, 'requests_response_text': AWS_SES_GOOD_RESPONSE, 'notify_response': False}), ('ses://user@example.com/T1JJ3T3L2/A1BRTD4JD/TIiacevi7FQ/us-west-2/user2@example.com', {'instance': NotifySES, 'requests_response_text': AWS_SES_GOOD_RESPONSE, 'response': False, 'requests_response_code': 999}), ('ses://user@example.com/T1JJ3T3L2/A1BRTD4JD/TIiajkdnlavi7FQ/us-west-2/user2@example.com', {'instance': NotifySES, 'requests_response_text': AWS_SES_GOOD_RESPONSE, 'test_requests_exceptions': True}))

def test_plugin_ses_urls():
    if False:
        print('Hello World!')
    '\n    NotifySES() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_ses_edge_cases(mock_post):
    if False:
        return 10
    '\n    NotifySES() Edge Cases\n\n    '
    with pytest.raises(TypeError):
        NotifySES(from_addr='user@example.eu', access_key_id=None, secret_access_key=TEST_ACCESS_KEY_SECRET, region_name=TEST_REGION, targets='user@example.ca')
    with pytest.raises(TypeError):
        NotifySES(from_addr='user@example.eu', access_key_id=TEST_ACCESS_KEY_ID, secret_access_key=None, region_name=TEST_REGION, targets='user@example.ca')
    with pytest.raises(TypeError):
        NotifySES(from_addr='user@example.eu', access_key_id=TEST_ACCESS_KEY_ID, secret_access_key=TEST_ACCESS_KEY_SECRET, region_name=None, targets='user@example.ca')
    obj = NotifySES(from_addr='user@example.eu', access_key_id=TEST_ACCESS_KEY_ID, secret_access_key=TEST_ACCESS_KEY_SECRET, region_name=TEST_REGION, targets=None)
    assert obj.notify(body='test', title='test') is False
    obj = NotifySES(from_addr='user@example.eu', access_key_id=TEST_ACCESS_KEY_ID, secret_access_key=TEST_ACCESS_KEY_SECRET, region_name=TEST_REGION, targets='invalid-email')
    assert obj.notify(body='test', title='test') is False

def test_plugin_ses_url_parsing():
    if False:
        return 10
    '\n    NotifySES() URL Parsing\n\n    '
    results = NotifySES.parse_url('ses://%s/%s/%s/%s/' % ('user@example.com', TEST_ACCESS_KEY_ID, TEST_ACCESS_KEY_SECRET, TEST_REGION))
    assert len(results['targets']) == 0
    assert 'region_name' in results
    assert TEST_REGION == results['region_name']
    assert 'access_key_id' in results
    assert TEST_ACCESS_KEY_ID == results['access_key_id']
    assert 'secret_access_key' in results
    assert TEST_ACCESS_KEY_SECRET == results['secret_access_key']
    results = NotifySES.parse_url('ses://%s/%s/%s/%s/%s/%s/' % ('user@example.com', TEST_ACCESS_KEY_ID, TEST_ACCESS_KEY_SECRET, TEST_REGION.upper(), 'user1@example.ca', 'user2@example.eu'))
    assert len(results['targets']) == 2
    assert 'user1@example.ca' in results['targets']
    assert 'user2@example.eu' in results['targets']
    assert 'region_name' in results
    assert TEST_REGION == results['region_name']
    assert 'access_key_id' in results
    assert TEST_ACCESS_KEY_ID == results['access_key_id']
    assert 'secret_access_key' in results
    assert TEST_ACCESS_KEY_SECRET == results['secret_access_key']

def test_plugin_ses_aws_response_handling():
    if False:
        return 10
    '\n    NotifySES() AWS Response Handling\n\n    '
    response = NotifySES.aws_response_to_dict(None)
    assert response['type'] is None
    assert response['request_id'] is None
    response = NotifySES.aws_response_to_dict('<Bad Response xmlns="http://ses.amazonaws.com/doc/2010-03-31/">')
    assert response['type'] is None
    assert response['request_id'] is None
    response = NotifySES.aws_response_to_dict('<SingleElement></SingleElement>')
    assert response['type'] == 'SingleElement'
    assert response['request_id'] is None
    response = NotifySES.aws_response_to_dict('')
    assert response['type'] is None
    assert response['request_id'] is None
    response = NotifySES.aws_response_to_dict('\n        <SendRawEmailResponse\n             xmlns="http://ses.amazonaws.com/doc/2010-12-01/">\n          <SendRawEmailResult>\n            <MessageId>\n               010f017d87656ee2-a2ea291f-79ea-44f3-9d25-00d041de307</MessageId>\n          </SendRawEmailResult>\n          <ResponseMetadata>\n            <RequestId>7abb454e-904b-4e46-a23c-2f4d2fc127a6</RequestId>\n          </ResponseMetadata>\n        </SendRawEmailResponse>\n        ')
    assert response['type'] == 'SendRawEmailResponse'
    assert response['request_id'] == '7abb454e-904b-4e46-a23c-2f4d2fc127a6'
    assert response['message_id'] == '010f017d87656ee2-a2ea291f-79ea-44f3-9d25-00d041de307'
    response = NotifySES.aws_response_to_dict('\n        <ErrorResponse xmlns="http://ses.amazonaws.com/doc/2010-03-31/">\n            <Error>\n                <Type>Sender</Type>\n                <Code>InvalidParameter</Code>\n                <Message>Invalid parameter</Message>\n            </Error>\n            <RequestId>b5614883-babe-56ca-93b2-1c592ba6191e</RequestId>\n        </ErrorResponse>\n        ')
    assert response['type'] == 'ErrorResponse'
    assert response['request_id'] == 'b5614883-babe-56ca-93b2-1c592ba6191e'
    assert response['error_type'] == 'Sender'
    assert response['error_code'] == 'InvalidParameter'
    assert response['error_message'] == 'Invalid parameter'

@mock.patch('requests.post')
def test_plugin_ses_attachments(mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifySES() Attachment Checks\n\n    '
    response = mock.Mock()
    response.content = AWS_SES_GOOD_RESPONSE
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    attach = AppriseAttachment(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    obj = Apprise.instantiate('ses://%s/%s/%s/%s/' % ('user@example.com', TEST_ACCESS_KEY_ID, TEST_ACCESS_KEY_SECRET, TEST_REGION))
    assert obj.notify(body='test', attach=attach) is True
    mock_post.reset_mock()
    attach.add(os.path.join(TEST_VAR_DIR, 'apprise-test.gif'))
    assert obj.notify(body='test', attach=attach) is True
    assert mock_post.call_count == 1
    mock_post.reset_mock()
    path = os.path.join(TEST_VAR_DIR, '/invalid/path/to/an/invalid/file.jpg')
    attach = AppriseAttachment(path)
    assert obj.notify(body='test', attach=attach) is False