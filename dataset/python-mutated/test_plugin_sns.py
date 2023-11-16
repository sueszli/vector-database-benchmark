from unittest import mock
import pytest
import requests
from apprise import Apprise
from apprise.plugins.NotifySNS import NotifySNS
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
TEST_ACCESS_KEY_ID = 'AHIAJGNT76XIMXDBIJYA'
TEST_ACCESS_KEY_SECRET = 'bu1dHSdO22pfaaVy/wmNsdljF4C07D3bndi9PQJ9'
TEST_REGION = 'us-east-2'
apprise_url_tests = (('sns://', {'instance': TypeError}), ('sns://:@/', {'instance': TypeError}), ('sns://T1JJ3T3L2', {'instance': TypeError}), ('sns://T1JJ3TD4JD/TIiajkdnlazk7FQ/', {'instance': TypeError}), ('sns://T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcevi7FQ/us-west-2/12223334444', {'instance': NotifySNS}), ('sns://?access=T1JJ3T3L2&secret=A1BRTD4JD/TIiajkdnlazkcevi7FQ&region=us-west-2&to=12223334444', {'instance': NotifySNS}), ('sns://T1JJ3TD4JD/TIiajkdnlazk7FQ/us-west-2/12223334444/12223334445', {'instance': NotifySNS, 'privacy_url': 'sns://T...D/****/us-west-2'}), ('sns://T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcOXrIdevi7FQ/us-east-1?to=12223334444', {'instance': NotifySNS}), ('sns://T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcevi7FQ/us-west-2/12223334444', {'instance': NotifySNS, 'response': False, 'requests_response_code': 999}), ('sns://T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkcevi7FQ/us-west-2/15556667777', {'instance': NotifySNS, 'test_requests_exceptions': True}))

def test_plugin_sns_urls():
    if False:
        print('Hello World!')
    '\n    NotifySNS() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_sns_edge_cases(mock_post):
    if False:
        while True:
            i = 10
    '\n    NotifySNS() Edge Cases\n\n    '
    with pytest.raises(TypeError):
        NotifySNS(access_key_id=None, secret_access_key=TEST_ACCESS_KEY_SECRET, region_name=TEST_REGION, targets='+1800555999')
    with pytest.raises(TypeError):
        NotifySNS(access_key_id=TEST_ACCESS_KEY_ID, secret_access_key=None, region_name=TEST_REGION, targets='+1800555999')
    with pytest.raises(TypeError):
        NotifySNS(access_key_id=TEST_ACCESS_KEY_ID, secret_access_key=TEST_ACCESS_KEY_SECRET, region_name=None, targets='+1800555999')
    obj = NotifySNS(access_key_id=TEST_ACCESS_KEY_ID, secret_access_key=TEST_ACCESS_KEY_SECRET, region_name=TEST_REGION, targets=None)
    assert obj.notify(body='test', title='test') is False
    obj = NotifySNS(access_key_id=TEST_ACCESS_KEY_ID, secret_access_key=TEST_ACCESS_KEY_SECRET, region_name=TEST_REGION, targets='+1809')
    assert obj.notify(body='test', title='test') is False
    obj = NotifySNS(access_key_id=TEST_ACCESS_KEY_ID, secret_access_key=TEST_ACCESS_KEY_SECRET, region_name=TEST_REGION, targets='#(invalid-topic-because-of-the-brackets)')
    assert obj.notify(body='test', title='test') is False

def test_plugin_sns_url_parsing():
    if False:
        while True:
            i = 10
    '\n    NotifySNS() URL Parsing\n\n    '
    results = NotifySNS.parse_url('sns://%s/%s/%s/' % (TEST_ACCESS_KEY_ID, TEST_ACCESS_KEY_SECRET, TEST_REGION))
    assert len(results['targets']) == 0
    assert 'region_name' in results
    assert TEST_REGION == results['region_name']
    assert 'access_key_id' in results
    assert TEST_ACCESS_KEY_ID == results['access_key_id']
    assert 'secret_access_key' in results
    assert TEST_ACCESS_KEY_SECRET == results['secret_access_key']
    results = NotifySNS.parse_url('sns://%s/%s/%s/%s/%s/' % (TEST_ACCESS_KEY_ID, TEST_ACCESS_KEY_SECRET, TEST_REGION.upper(), '+18001234567', 'MyTopic'))
    assert len(results['targets']) == 2
    assert '+18001234567' in results['targets']
    assert 'MyTopic' in results['targets']
    assert 'region_name' in results
    assert TEST_REGION == results['region_name']
    assert 'access_key_id' in results
    assert TEST_ACCESS_KEY_ID == results['access_key_id']
    assert 'secret_access_key' in results
    assert TEST_ACCESS_KEY_SECRET == results['secret_access_key']

def test_plugin_sns_object_parsing():
    if False:
        i = 10
        return i + 15
    '\n    NotifySNS() Object Parsing\n\n    '
    a = Apprise()
    assert a.add('sns://') is False
    assert a.add('sns://nosecret') is False
    assert a.add('sns://nosecret/noregion/') is False
    assert a.add('sns://norecipient/norecipient/us-west-2') is True
    assert len(a) == 1
    assert a.add('sns://oh/yeah/us-west-2/abcdtopic/+12223334444') is True
    assert len(a) == 2
    assert a.add('sns://oh/yeah/us-west-2/12223334444') is True
    assert len(a) == 3

def test_plugin_sns_aws_response_handling():
    if False:
        print('Hello World!')
    '\n    NotifySNS() AWS Response Handling\n\n    '
    response = NotifySNS.aws_response_to_dict(None)
    assert response['type'] is None
    assert response['request_id'] is None
    response = NotifySNS.aws_response_to_dict('<Bad Response xmlns="http://sns.amazonaws.com/doc/2010-03-31/">')
    assert response['type'] is None
    assert response['request_id'] is None
    response = NotifySNS.aws_response_to_dict('<SingleElement></SingleElement>')
    assert response['type'] == 'SingleElement'
    assert response['request_id'] is None
    response = NotifySNS.aws_response_to_dict('')
    assert response['type'] is None
    assert response['request_id'] is None
    response = NotifySNS.aws_response_to_dict('\n        <PublishResponse xmlns="http://sns.amazonaws.com/doc/2010-03-31/">\n            <PublishResult>\n                <MessageId>5e16935a-d1fb-5a31-a716-c7805e5c1d2e</MessageId>\n            </PublishResult>\n            <ResponseMetadata>\n                <RequestId>dc258024-d0e6-56bb-af1b-d4fe5f4181a4</RequestId>\n            </ResponseMetadata>\n        </PublishResponse>\n        ')
    assert response['type'] == 'PublishResponse'
    assert response['request_id'] == 'dc258024-d0e6-56bb-af1b-d4fe5f4181a4'
    assert response['message_id'] == '5e16935a-d1fb-5a31-a716-c7805e5c1d2e'
    response = NotifySNS.aws_response_to_dict('\n         <CreateTopicResponse xmlns="http://sns.amazonaws.com/doc/2010-03-31/">\n           <CreateTopicResult>\n             <TopicArn>arn:aws:sns:us-east-1:000000000000:abcd</TopicArn>\n                </CreateTopicResult>\n            <ResponseMetadata>\n                <RequestId>604bef0f-369c-50c5-a7a4-bbd474c83d6a</RequestId>\n            </ResponseMetadata>\n        </CreateTopicResponse>\n        ')
    assert response['type'] == 'CreateTopicResponse'
    assert response['request_id'] == '604bef0f-369c-50c5-a7a4-bbd474c83d6a'
    assert response['topic_arn'] == 'arn:aws:sns:us-east-1:000000000000:abcd'
    response = NotifySNS.aws_response_to_dict('\n        <ErrorResponse xmlns="http://sns.amazonaws.com/doc/2010-03-31/">\n            <Error>\n                <Type>Sender</Type>\n                <Code>InvalidParameter</Code>\n                <Message>Invalid parameter: TopicArn or TargetArn Reason:\n                no value for required parameter</Message>\n            </Error>\n            <RequestId>b5614883-babe-56ca-93b2-1c592ba6191e</RequestId>\n        </ErrorResponse>\n        ')
    assert response['type'] == 'ErrorResponse'
    assert response['request_id'] == 'b5614883-babe-56ca-93b2-1c592ba6191e'
    assert response['error_type'] == 'Sender'
    assert response['error_code'] == 'InvalidParameter'
    assert response['error_message'].startswith('Invalid parameter:')
    assert response['error_message'].endswith('required parameter')

@mock.patch('requests.post')
def test_plugin_sns_aws_topic_handling(mock_post):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifySNS() AWS Topic Handling\n\n    '
    arn_response = '\n         <CreateTopicResponse xmlns="http://sns.amazonaws.com/doc/2010-03-31/">\n           <CreateTopicResult>\n             <TopicArn>arn:aws:sns:us-east-1:000000000000:abcd</TopicArn>\n                </CreateTopicResult>\n            <ResponseMetadata>\n                <RequestId>604bef0f-369c-50c5-a7a4-bbd474c83d6a</RequestId>\n            </ResponseMetadata>\n        </CreateTopicResponse>\n        '

    def post(url, data, **kwargs):
        if False:
            print('Hello World!')
        "\n        Since Publishing a token requires 2 posts, we need to return our\n        response depending on what step we're on\n        "
        robj = mock.Mock()
        robj.text = ''
        robj.status_code = requests.codes.ok
        if data.find('=CreateTopic') >= 0:
            robj.status_code = requests.codes.bad_request
        return robj
    mock_post.side_effect = post
    a = Apprise()
    a.add(['sns://T1JJ3T3L2/A1BRTD4JD/TIiajkdnl/us-west-2/TopicA', 'sns://T1JJ3T3L2/A1BRTD4JD/TIiajkdnl/us-east-1/TopicA/TopicB/sns://T1JJ3T3L2/A1BRTD4JD/TIiajkdnlazkce/us-west-2/12223334444/TopicA'])
    assert a.notify(title='', body='test') is False

    def post(url, data, **kwargs):
        if False:
            print('Hello World!')
        "\n        Since Publishing a token requires 2 posts, we need to return our\n        response depending on what step we're on\n        "
        robj = mock.Mock()
        robj.text = ''
        robj.status_code = requests.codes.ok
        if data.find('=CreateTopic') >= 0:
            robj.text = arn_response
        elif data.find('=Publish') >= 0 and data.find('TopicArn=') >= 0:
            robj.status_code = requests.codes.bad_request
        return robj
    mock_post.side_effect = post
    assert a.notify(title='', body='test') is False
    mock_post.side_effect = None
    robj = mock.Mock()
    robj.text = '<CreateTopicResponse></CreateTopicResponse>'
    robj.status_code = requests.codes.ok
    mock_post.return_value = robj
    assert a.notify(title='', body='test') is False
    robj = mock.Mock()
    robj.text = ''
    robj.status_code = requests.codes.bad_request
    mock_post.return_value = robj
    assert a.notify(title='', body='test') is False
    robj = mock.Mock()
    robj.text = arn_response
    robj.status_code = requests.codes.ok
    mock_post.return_value = robj
    assert a.notify(title='', body='test') is True