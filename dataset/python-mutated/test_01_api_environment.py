"""Test Apis."""
import urllib
import pytest
from .factories import TestApiBase
from .utils import *

@pytest.mark.usefixtures('db')
class TestApiEnv(TestApiBase):
    """api role testing"""
    uri_prefix = '/api/environment'
    env_id = {}
    env_data = {'env_name': u'测试环境', 'space_id': 1}
    user_name_2 = u'Production'
    env_data_2 = {'env_name': u'Production', 'space_id': 1}
    env_data_remove = {'env_name': u'environment_remove', 'space_id': 1}

    def test_init(self, user, testapp, client, db):
        if False:
            i = 10
            return i + 15
        self.init_vars(self.env_data)
        self.init_vars(self.env_data_2)
        self.init_vars(self.env_data_remove)

    def test_create(self, user, testapp, client, db):
        if False:
            return 10
        'create successful.'
        resp = client.post('%s/' % self.uri_prefix, data=self.env_data)
        response_success(resp)
        compare_req_resp(self.env_data, resp)
        self.env_data['id'] = resp_json(resp)['data']['id']
        resp = client.post('%s/' % self.uri_prefix, data=self.env_data_2)
        response_success(resp)
        compare_req_resp(self.env_data_2, resp)
        self.env_data_2['id'] = resp_json(resp)['data']['id']

    def test_one(self, user, testapp, client, db):
        if False:
            print('Hello World!')
        'item successful.'
        resp = client.get('%s/%d' % (self.uri_prefix, self.env_data['id']))
        response_success(resp)
        compare_req_resp(self.env_data, resp)

    def test_get_list_page_size(self, user, testapp, client):
        if False:
            print('Hello World!')
        'test list should create 2 users at least, due to test pagination, searching.'
        query = {'page': 1, 'size': 1}
        response = {'count': 2}
        resp = client.get('%s/?%s' % (self.uri_prefix, urlencode(query)))
        response_success(resp)
        resp_dict = resp_json(resp)
        compare_in(self.env_data_2, resp_dict['data']['list'].pop())
        compare_req_resp(response, resp)

    def test_get_list_query(self, user, testapp, client):
        if False:
            i = 10
            return i + 15
        'test list should create 2 users at least, due to test pagination, searching.'
        query = {'page': 1, 'size': 1, 'kw': self.user_name_2}
        response = {'count': 1}
        resp = client.get('%s/?%s' % (self.uri_prefix, urlencode(query)))
        response_success(resp)
        resp_dict = resp_json(resp)
        compare_in(self.env_data_2, resp_dict['data']['list'].pop())
        compare_req_resp(response, resp)

    def test_get_update(self, user, testapp, client):
        if False:
            while True:
                i = 10
        'Login successful.'
        env_data_2 = self.env_data_2
        env_data_2['env_name'] = 'Tester_edit'
        resp = client.put('%s/%d' % (self.uri_prefix, self.env_data_2['id']), data=env_data_2)
        response_success(resp)
        compare_req_resp(env_data_2, resp)
        resp = client.get('%s/%d' % (self.uri_prefix, self.env_data_2['id']))
        response_success(resp)
        compare_req_resp(env_data_2, resp)

    def test_get_remove(self, user, testapp, client):
        if False:
            for i in range(10):
                print('nop')
        'Login successful.'
        resp = client.post('%s/' % self.uri_prefix, data=self.env_data_remove)
        env_id = resp_json(resp)['data']['id']
        response_success(resp)
        resp = client.delete('%s/%d' % (self.uri_prefix, env_id))
        response_success(resp)
        resp = client.get('%s/%d' % (self.uri_prefix, env_id))
        response_error(resp)