"""Test Apis."""
import urllib
import pytest
from .factories import TestApiBase
from .utils import *

@pytest.mark.usefixtures('db')
class TestApiServer(TestApiBase):
    """api role testing"""
    uri_prefix = '/api/server'
    server_id = {}
    server_data = {'name': u'开发机01', 'host': u'127.0.0.1', 'user': u'work', 'port': 22}
    server_name_2 = u'test02'
    server_data_2 = {'name': u'test02', 'host': u'192.168.0.1', 'user': u'work', 'port': 22}
    server_data_remove = {'name': u'this server will be deleted soon', 'host': u'11.22.33.44', 'user': u'work', 'port': 22}

    def test_create(self, user, testapp, client, db):
        if False:
            i = 10
            return i + 15
        'create successful.'
        resp = client.post('%s/' % self.uri_prefix, data=self.server_data)
        response_success(resp)
        compare_req_resp(self.server_data, resp)
        self.server_data['id'] = resp_json(resp)['data']['id']
        resp = client.post('%s/' % self.uri_prefix, data=self.server_data_2)
        response_success(resp)
        compare_req_resp(self.server_data_2, resp)
        self.server_data_2['id'] = resp_json(resp)['data']['id']

    def test_one(self, user, testapp, client, db):
        if False:
            while True:
                i = 10
        'item successful.'
        resp = client.get('%s/%d' % (self.uri_prefix, self.server_data['id']))
        response_success(resp)
        compare_req_resp(self.server_data, resp)

    def test_get_list_page_size(self, user, testapp, client):
        if False:
            for i in range(10):
                print('nop')
        'test list should create 2 users at least, due to test pagination, searching.'
        query = {'page': 1, 'size': 1}
        response = {'count': 2}
        resp = client.get('%s/?%s' % (self.uri_prefix, urlencode(query)))
        response_success(resp)
        resp_dict = resp_json(resp)
        compare_in(self.server_data_2, resp_dict['data']['list'].pop())
        compare_req_resp(response, resp)

    def test_get_list_query(self, user, testapp, client):
        if False:
            i = 10
            return i + 15
        'test list should create 2 users at least, due to test pagination, searching.'
        query = {'page': 1, 'size': 1, 'kw': self.server_name_2}
        response = {'count': 1}
        resp = client.get('%s/?%s' % (self.uri_prefix, urlencode(query)))
        response_success(resp)
        resp_dict = resp_json(resp)
        compare_in(self.server_data_2, resp_dict['data']['list'].pop())
        compare_req_resp(response, resp)

    def test_get_update(self, user, testapp, client):
        if False:
            for i in range(10):
                print('nop')
        'Login successful.'
        server_data_2 = self.server_data_2
        server_data_2['name'] = 'Tester_edit'
        resp = client.put('%s/%d' % (self.uri_prefix, self.server_data_2['id']), data=server_data_2)
        response_success(resp)
        compare_req_resp(server_data_2, resp)
        resp = client.get('%s/%d' % (self.uri_prefix, self.server_data_2['id']))
        response_success(resp)
        compare_req_resp(server_data_2, resp)

    def test_get_remove(self, user, testapp, client):
        if False:
            print('Hello World!')
        'Login successful.'
        resp = client.post('%s/' % self.uri_prefix, data=self.server_data_remove)
        server_id = resp_json(resp)['data']['id']
        response_success(resp)
        resp = client.delete('%s/%d' % (self.uri_prefix, server_id))
        response_success(resp)
        resp = client.get('%s/%d' % (self.uri_prefix, server_id))
        response_error(resp)