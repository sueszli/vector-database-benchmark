"""Test Apis."""
from copy import deepcopy
import pytest
from flask import current_app
from .factories import TestApiBase
from .test_00_base import space_base
from .utils import *

@pytest.mark.usefixtures('db')
class TestApiSpace(TestApiBase):
    """api role testing"""
    uri_prefix = '/api/space'
    user_id = {}
    space_default_base = deepcopy(space_base)
    space_data = {'name': u'大数据', 'user_id': u'1', 'members': json.dumps([{'user_id': 2, 'role': 'MASTER'}, {'user_id': 3, 'role': 'DEVELOPER'}])}
    space_name_2 = u'瓦力'
    space_data_2 = {'name': u'瓦力', 'user_id': u'2', 'members': json.dumps([{'user_id': 3, 'role': 'MASTER'}, {'user_id': 1, 'role': 'DEVELOPER'}])}
    space_data_remove = {'name': u'瓦尔登', 'user_id': u'2', 'members': json.dumps([{'user_id': 1, 'role': 'MASTER'}, {'user_id': 3, 'role': 'DEVELOPER'}])}

    def test_get_update_default_space(self, user, testapp, client):
        if False:
            i = 10
            return i + 15
        'Login successful.'
        self.space_default_base['members'] = json.dumps([{'user_id': 2, 'role': 'MASTER'}, {'user_id': 3, 'role': 'DEVELOPER'}])
        resp = client.put('%s/%d' % (self.uri_prefix, 1), data=self.space_default_base)
        response_success(resp)
        self.compare_member_req_resp(self.space_data, resp)

    def test_create(self, user, testapp, client, db):
        if False:
            print('Hello World!')
        'create successful.'
        resp = client.post('%s/' % self.uri_prefix, data=self.space_data)
        response_success(resp)
        current_app.logger.info(resp_json(resp)['data'])
        self.compare_member_req_resp(self.space_data, resp)
        self.space_data['space_id'] = resp_json(resp)['data']['id']
        'create successful.'
        resp = client.post('%s/' % self.uri_prefix, data=self.space_data_2)
        response_success(resp)
        self.compare_member_req_resp(self.space_data_2, resp)
        self.space_data_2['space_id'] = resp_json(resp)['data']['id']

    def test_one(self, user, testapp, client, db):
        if False:
            i = 10
            return i + 15
        'item successful.'
        resp = client.get('%s/%d' % (self.uri_prefix, self.space_data['space_id']))
        response_success(resp)
        self.compare_member_req_resp(self.space_data, resp)

    def test_get_update(self, user, testapp, client):
        if False:
            while True:
                i = 10
        'Login successful.'
        space_data = self.space_data
        space_data['name'] = u'大数据平台'
        resp = client.put('%s/%d' % (self.uri_prefix, self.space_data['space_id']), data=space_data)
        response_success(resp)
        self.compare_member_req_resp(self.space_data, resp)
        space_data_2 = self.space_data_2
        space_data_2['name'] = u'瓦力2.0'
        resp = client.put('%s/%d' % (self.uri_prefix, self.space_data_2['space_id']), data=space_data_2)
        response_success(resp)
        self.compare_member_req_resp(self.space_data_2, resp)
        resp = client.get('%s/%d' % (self.uri_prefix, self.space_data_2['space_id']))
        response_success(resp)
        response_success(resp)
        self.compare_member_req_resp(self.space_data_2, resp)

    def compare_member_req_resp(self, request, response):
        if False:
            print('Hello World!')
        for user_response in resp_json(response)['data']['members']:
            for user_request in json.loads(request['members']):
                if user_request['user_id'] == user_response['user_id']:
                    assert user_request['role'] == user_response['role']