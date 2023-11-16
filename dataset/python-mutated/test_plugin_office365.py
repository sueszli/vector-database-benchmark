import os
from unittest import mock
import pytest
import requests
from datetime import datetime
from json import dumps
from apprise import Apprise
from apprise.plugins.NotifyOffice365 import NotifyOffice365
from helpers import AppriseURLTester
import logging
logging.disable(logging.CRITICAL)
TEST_VAR_DIR = os.path.join(os.path.dirname(__file__), 'var')
apprise_url_tests = (('o365://', {'instance': TypeError}), ('o365://:@/', {'instance': TypeError}), ('o365://{tenant}:{aid}/{cid}/{secret}/{targets}'.format(tenant=',', cid='ab-cd-ef-gh', aid='user@example.com', secret='abcd/123/3343/@jack/test', targets='/'.join(['email1@test.ca'])), {'instance': TypeError}), ('o365://{tenant}:{aid}/{cid}/{secret}/{targets}'.format(tenant='tenant', cid='ab.', aid='user@example.com', secret='abcd/123/3343/@jack/test', targets='/'.join(['email1@test.ca'])), {'instance': TypeError}), ('o365://{tenant}:{aid}/{cid}/{secret}/{targets}'.format(tenant='tenant', cid='ab-cd-ef-gh', aid='user@example.com', secret='abcd/123/3343/@jack/test', targets='/'.join(['email1@test.ca'])), {'instance': NotifyOffice365, 'requests_response_text': {'expires_in': 2000, 'access_token': 'abcd1234'}, 'privacy_url': 'o365://t...t:user@example.com/a...h/****/email1%40test.ca/'}), ('o365://_/?oauth_id={cid}&oauth_secret={secret}&tenant={tenant}&to={targets}&from={aid}'.format(tenant='tenant', cid='ab-cd-ef-gh', aid='user@example.com', secret='abcd/123/3343/@jack/test', targets='email1@test.ca'), {'instance': NotifyOffice365, 'requests_response_text': {'expires_in': 2000, 'access_token': 'abcd1234'}, 'privacy_url': 'o365://t...t:user@example.com/a...h/****/email1%40test.ca/'}), ('o365://{tenant}:{aid}/{cid}/{secret}/{targets}'.format(tenant='tenant', cid='ab-cd-ef-gh', aid='user@example.com', secret='abcd/123/3343/@jack/test', targets='/'.join(['email1@test.ca'])), {'instance': NotifyOffice365, 'requests_response_text': '{', 'notify_response': False}), ('o365://{tenant}:{aid}/{cid}/{secret}'.format(tenant='tenant', cid='ab-cd-ef-gh', aid='user@example.com', secret='abcd/123/3343/@jack/test'), {'instance': NotifyOffice365, 'requests_response_text': {'expires_in': 2000, 'access_token': 'abcd1234'}}), ('o365://{tenant}:{aid}/{cid}/{secret}/{targets}'.format(tenant='tenant', cid='zz-zz-zz-zz', aid='user@example.com', secret='abcd/abc/dcba/@john/test', targets='/'.join(['email1@test.ca'])), {'instance': NotifyOffice365, 'response': False, 'requests_response_code': 999}), ('o365://{tenant}:{aid}/{cid}/{secret}/{targets}'.format(tenant='tenant', cid='01-12-23-34', aid='user@example.com', secret='abcd/321/4321/@test/test', targets='/'.join(['email1@test.ca'])), {'instance': NotifyOffice365, 'test_requests_exceptions': True}))

def test_plugin_office365_urls():
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyOffice365() Apprise URLs\n\n    '
    AppriseURLTester(tests=apprise_url_tests).run_all()

@mock.patch('requests.post')
def test_plugin_office365_general(mock_post):
    if False:
        for i in range(10):
            print('nop')
    '\n    NotifyOffice365() General Testing\n\n    '
    email = 'user@example.net'
    tenant = 'ff-gg-hh-ii-jj'
    client_id = 'aa-bb-cc-dd-ee'
    secret = 'abcd/1234/abcd@ajd@/test'
    targets = 'target@example.com'
    authentication = {'token_type': 'Bearer', 'expires_in': 6000, 'access_token': 'abcd1234'}
    response = mock.Mock()
    response.content = dumps(authentication)
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    obj = Apprise.instantiate('o365://{tenant}:{email}/{tenant}/{secret}/{targets}'.format(tenant=tenant, email=email, secret=secret, targets=targets))
    assert isinstance(obj, NotifyOffice365)
    assert isinstance(obj.url(), str)
    assert obj.notify(title='title', body='test') is True
    obj = Apprise.instantiate('o365://{tenant}:{email}/{tenant}/{secret}/{targets}?bcc={bcc}&cc={cc}'.format(tenant=tenant, email=email, secret=secret, targets=targets, cc='Chuck Norris cnorris@yahoo.ca, Sauron@lotr.me, invalid@!', bcc='Bruce Willis bwillis@hotmail.com, Frodo@lotr.me invalid@!'))
    assert isinstance(obj, NotifyOffice365)
    assert isinstance(obj.url(), str)
    assert obj.notify(title='title', body='test') is True
    with pytest.raises(TypeError):
        NotifyOffice365(email=email, client_id=client_id, tenant=tenant, secret=None, targets=None)
    with pytest.raises(TypeError):
        NotifyOffice365(email=None, client_id=client_id, tenant=tenant, secret=secret, targets=None)
    with pytest.raises(TypeError):
        NotifyOffice365(email='garbage', client_id=client_id, tenant=tenant, secret=secret, targets=None)
    obj = NotifyOffice365(email=email, client_id=client_id, tenant=tenant, secret=secret, targets=('Management abc@gmail.com', 'garbage'))
    assert obj.notify(title='title', body='test') is True
    obj = NotifyOffice365(email=email, client_id=client_id, tenant=tenant, secret=secret, targets=('invalid', 'garbage'))
    assert obj.notify(title='title', body='test') is False

@mock.patch('requests.post')
def test_plugin_office365_authentication(mock_post):
    if False:
        print('Hello World!')
    '\n    NotifyOffice365() Authentication Testing\n\n    '
    tenant = 'ff-gg-hh-ii-jj'
    email = 'user@example.net'
    client_id = 'aa-bb-cc-dd-ee'
    secret = 'abcd/1234/abcd@ajd@/test'
    targets = 'target@example.com'
    authentication_okay = {'token_type': 'Bearer', 'expires_in': 6000, 'access_token': 'abcd1234'}
    authentication_failure = {'error': 'invalid_scope', 'error_description': 'AADSTS70011: Blah... Blah Blah... Blah', 'error_codes': [70011], 'timestamp': '2020-01-09 02:02:12Z', 'trace_id': '255d1aef-8c98-452f-ac51-23d051240864', 'correlation_id': 'fb3d2015-bc17-4bb9-bb85-30c5cf1aaaa7'}
    response = mock.Mock()
    response.content = dumps(authentication_okay)
    response.status_code = requests.codes.ok
    mock_post.return_value = response
    obj = Apprise.instantiate('o365://{tenant}:{email}/{client_id}/{secret}/{targets}'.format(client_id=client_id, tenant=tenant, email=email, secret=secret, targets=targets))
    assert isinstance(obj, NotifyOffice365)
    assert obj.authenticate() is True
    assert obj.authenticate() is True
    obj.token_expiry = datetime.now()
    assert obj.authenticate() is True
    response.status_code = 400
    assert obj.notify(title='title', body='test') is False
    obj.token_expiry = datetime.now()
    response.content = dumps(authentication_failure)
    assert obj.authenticate() is False
    assert obj.notify(title='title', body='test') is False
    invalid_auth_entries = authentication_okay.copy()
    invalid_auth_entries['expires_in'] = 'garbage'
    response.content = dumps(invalid_auth_entries)
    response.status_code = requests.codes.ok
    assert obj.authenticate() is False
    invalid_auth_entries['expires_in'] = None
    response.content = dumps(invalid_auth_entries)
    assert obj.authenticate() is False
    invalid_auth_entries['expires_in'] = ''
    response.content = dumps(invalid_auth_entries)
    assert obj.authenticate() is False
    del invalid_auth_entries['expires_in']
    response.content = dumps(invalid_auth_entries)
    assert obj.authenticate() is False