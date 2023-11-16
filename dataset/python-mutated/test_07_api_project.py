"""Test Apis."""
import urllib
import pytest
from flask import current_app
from .factories import TestApiBase
from .utils import *

@pytest.mark.usefixtures('db')
class TestApiProject(TestApiBase):
    """api role testing"""
    uri_prefix = '/api/project'
    server_id = {}
    project_data = {'environment_id': 1, 'space_id': 'to be init_vars', 'excludes': u'*.log', 'keep_version_num': 11, 'name': u'walden-瓦尔登', 'post_deploy': u'echo post_deploy', 'post_release': u'echo post_release', 'prev_deploy': u'echo prev_deploy', 'prev_release': u'echo prev_release', 'repo_mode': u'branch', 'repo_password': u'', 'repo_url': u'git@github.com:meolu/walle-web.git', 'repo_username': u'', 'server_ids': u'1,2', 'target_releases': u'/tmp/walle/library', 'target_root': u'/tmp/walle/root', 'task_vars': u'debug=1;\\napp=auotapp.py', 'user_id': 'to be init_vars'}
    project_data_members = [{'user_id': 3, 'role': 'MASTER'}, {'user_id': 2, 'role': 'DEVELOPER'}]
    project_data_members_error = [{'user_id': 3, 'role': 'MASTER'}, {'user_id': 2, 'role': 'DEVELOPER'}, {'user_id': 4, 'role': 'DEVELOPER'}]
    project_name_2 = u'walle-web'
    project_data_2 = {'environment_id': 2, 'space_id': 'to be init_vars', 'excludes': u'*.log', 'keep_version_num': 10, 'name': u'walle-web', 'post_deploy': u'echo post_deploy', 'post_release': u'echo post_release', 'prev_deploy': u'echo prev_deploy', 'prev_release': u'echo prev_release', 'repo_mode': u'branch', 'repo_password': u'', 'repo_url': u'git@github.com:meolu/walle-web.git', 'repo_username': u'', 'server_ids': u'1,2', 'target_releases': u'/tmp/walle/library', 'target_root': u'/tmp/walle/root', 'task_vars': u'debug=1;\\napp=auotapp.py', 'user_id': 'to be init_vars'}
    project_data_2_update = {'environment_id': 1, 'space_id': 'to be init_vars', 'excludes': u'*.log', 'keep_version_num': 11, 'name': u'walle-web to walden edit', 'post_deploy': u'echo post_deploy; pwd', 'post_release': u'echo post_release; pwd', 'prev_deploy': u'echo prev_deploy; pwd', 'prev_release': u'echo prev_release; pwd', 'repo_mode': u'tag', 'repo_password': u'', 'repo_url': u'git@github.com:meolu/walden.git', 'repo_username': u'', 'server_ids': u'1,2', 'target_releases': u'/tmp/walden/library', 'target_root': u'/tmp/walden/root', 'task_vars': u'debug=1;\\napp=auotapp.py; project=walden', 'user_id': 'to be init_vars'}
    project_data_remove = {'name': u'this server will be deleted soon', 'environment_id': 1, 'space_id': 'to be init_vars', 'excludes': u'*.log', 'keep_version_num': 11, 'post_deploy': u'echo post_deploy', 'post_release': u'echo post_release', 'prev_deploy': u'echo prev_deploy', 'prev_release': u'echo prev_release', 'repo_mode': u'branch', 'repo_password': u'', 'repo_url': u'git@github.com:meolu/walle-web.git', 'repo_username': u'', 'server_ids': u'1,2', 'target_releases': u'/tmp/walle/library', 'target_root': u'/tmp/walle/root', 'task_vars': u'debug=1;\\napp=auotapp.py', 'user_id': 'to be init_vars'}

    def test_init(self, user, testapp, client, db):
        if False:
            return 10
        self.init_vars(self.project_data)
        self.init_vars(self.project_data_2)
        self.init_vars(self.project_data_remove)
        self.init_vars(self.project_data_2_update)

    def test_create(self, user, testapp, client, db):
        if False:
            return 10
        'create successful.'
        project_data = dict(self.project_data, **dict({'members': json.dumps(self.project_data_members)}))
        resp = client.post('%s/' % self.uri_prefix, data=project_data)
        response_success(resp)
        self.project_compare_req_resp(self.project_data, resp)
        self.project_data['id'] = resp_json(resp)['data']['id']
        resp = client.post('%s/' % self.uri_prefix, data=self.project_data_2)
        response_success(resp)
        self.project_compare_req_resp(self.project_data_2, resp)
        self.project_data_2['id'] = resp_json(resp)['data']['id']

    def test_one(self, user, testapp, client, db):
        if False:
            i = 10
            return i + 15
        'item successful.'
        resp = client.get('%s/%d' % (self.uri_prefix, self.project_data['id']))
        response_success(resp)
        self.project_compare_req_resp(self.project_data, resp)

    def test_get_list_page_size(self, user, testapp, client):
        if False:
            return 10
        'test list should create 2 users at least, due to test pagination, searching.'
        query = {'page': 1, 'size': 1}
        response = {'count': 2}
        resp = client.get('%s/?%s' % (self.uri_prefix, urlencode(query)))
        response_success(resp)
        resp_dict = resp_json(resp)
        self.project_compare_in(self.project_data_2, resp_dict['data']['list'].pop())
        self.project_compare_req_resp(response, resp)

    def test_get_list_query(self, user, testapp, client):
        if False:
            for i in range(10):
                print('nop')
        'test list should create 2 users at least, due to test pagination, searching.'
        query = {'page': 1, 'size': 1, 'kw': self.project_name_2}
        response = {'count': 1}
        resp = client.get('%s/?%s' % (self.uri_prefix, urlencode(query)))
        response_success(resp)
        resp_dict = resp_json(resp)
        self.project_compare_in(self.project_data_2, resp_dict['data']['list'].pop())
        self.project_compare_req_resp(response, resp)

    def test_get_update(self, user, testapp, client):
        if False:
            for i in range(10):
                print('nop')
        'Login successful.'
        resp = client.put('%s/%d' % (self.uri_prefix, self.project_data_2['id']), data=self.project_data_2_update)
        response_success(resp)
        self.project_compare_req_resp(self.project_data_2_update, resp)
        resp = client.get('%s/%d' % (self.uri_prefix, self.project_data_2['id']))
        response_success(resp)
        self.project_compare_req_resp(self.project_data_2_update, resp)

    def test_get_update_members(self, user, testapp, client):
        if False:
            return 10
        'Login successful.'
        from walle.service.code import Code
        headers = {'content-type': 'application/json'}
        resp = client.put('%s/%d/members' % (self.uri_prefix, self.project_data_2['id']), data=json.dumps(self.project_data_members_error), headers=headers)
        current_app.logger.info(resp)
        response_error(resp, Code.user_not_in_space)
        headers = {'content-type': 'application/json'}
        resp = client.put('%s/%d/members' % (self.uri_prefix, self.project_data_2['id']), data=json.dumps(self.project_data_members), headers=headers)
        current_app.logger.info(resp)
        response_success(resp)
        current_app.logger.info(resp_json(resp)['data'])
        self.compare_member_req_resp_without_key(self.project_data_members, resp)
        resp = client.get('%s/%d' % (self.uri_prefix, self.project_data_2['id']))
        response_success(resp)
        self.project_data_2_update['members'] = json.dumps(self.project_data_members)
        self.compare_member_req_resp(self.project_data_2_update, resp)

    def test_get_remove(self, user, testapp, client):
        if False:
            print('Hello World!')
        'Login successful.'
        resp = client.post('%s/' % self.uri_prefix, data=self.project_data_remove)
        project_id = resp_json(resp)['data']['id']
        response_success(resp)
        resp = client.delete('%s/%d' % (self.uri_prefix, project_id))
        response_success(resp)
        resp = client.get('%s/%d' % (self.uri_prefix, project_id))
        response_error(resp)

    def get_list_ids(self, projectOrigin):
        if False:
            while True:
                i = 10
        group_list = projectOrigin.copy()
        group_list['user_ids'] = map(int, projectOrigin['user_ids'].split(','))
        return group_list

    def project_compare_req_resp(self, req_obj, resp):
        if False:
            while True:
                i = 10
        '\n        there is some thing difference in project api\n        such as server_ids\n        :param resp:\n        :return:\n        '
        resp_obj = resp_json(resp)['data']
        servers = []
        if 'server_info' in resp_obj:
            for server in resp_obj['server_info']:
                servers.append(str(server['id']))
        self.project_compare_in(req_obj, resp_obj)

    def project_compare_in(self, req_obj, resp_obj):
        if False:
            for i in range(10):
                print('nop')
        for (k, v) in req_obj.items():
            assert k in resp_obj.keys(), 'Key %r not in response (keys are %r)' % (k, resp_obj.keys())
            assert resp_obj[k] == v, 'Value for key %r should be %r but is %r' % (k, v, resp_obj[k])

    def compare_member_req_resp(self, request, response):
        if False:
            while True:
                i = 10
        for user_response in resp_json(response)['data']['members']:
            for user_request in json.loads(request['members']):
                if user_request['user_id'] == user_response['user_id']:
                    assert user_request['role'] == user_response['role']

    def compare_member_req_resp_without_key(self, request, response):
        if False:
            while True:
                i = 10
        for user_response in resp_json(response)['data']:
            for user_request in request:
                if user_request['user_id'] == user_response['user_id']:
                    assert user_request['role'] == user_response['role']