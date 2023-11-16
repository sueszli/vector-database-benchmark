"""Unit tests for Superset"""
from datetime import datetime, timedelta
import json
from typing import Optional
from unittest.mock import ANY
from flask_appbuilder.security.sqla.models import User
import prison
from unittest.mock import patch
from superset import db
from superset.models.core import Log
from superset.views.log.api import LogRestApi
from tests.integration_tests.conftest import with_feature_flags
from tests.integration_tests.dashboard_utils import create_dashboard
from tests.integration_tests.test_app import app
from .base_tests import SupersetTestCase
EXPECTED_COLUMNS = ['action', 'dashboard_id', 'dttm', 'duration_ms', 'json', 'referrer', 'slice_id', 'user', 'user_id']

class TestLogApi(SupersetTestCase):

    def insert_log(self, action: str, user: 'User', dashboard_id: Optional[int]=0, slice_id: Optional[int]=0, json: Optional[str]='', duration_ms: Optional[int]=0):
        if False:
            for i in range(10):
                print('nop')
        log = Log(action=action, user=user, dashboard_id=dashboard_id, slice_id=slice_id, json=json, duration_ms=duration_ms)
        db.session.add(log)
        db.session.commit()
        return log

    def test_not_enabled(self):
        if False:
            while True:
                i = 10
        with patch.object(LogRestApi, 'is_enabled', return_value=False):
            admin_user = self.get_user('admin')
            self.insert_log('some_action', admin_user)
            self.login(username='admin')
            arguments = {'filters': [{'col': 'action', 'opr': 'sw', 'value': 'some_'}]}
            uri = f'api/v1/log/?q={prison.dumps(arguments)}'
            rv = self.client.get(uri)
            self.assertEqual(rv.status_code, 404)

    def test_get_list(self):
        if False:
            return 10
        '\n        Log API: Test get list\n        '
        admin_user = self.get_user('admin')
        log = self.insert_log('some_action', admin_user)
        self.login(username='admin')
        arguments = {'filters': [{'col': 'action', 'opr': 'sw', 'value': 'some_'}]}
        uri = f'api/v1/log/?q={prison.dumps(arguments)}'
        rv = self.client.get(uri)
        self.assertEqual(rv.status_code, 200)
        response = json.loads(rv.data.decode('utf-8'))
        self.assertEqual(list(response['result'][0].keys()), EXPECTED_COLUMNS)
        self.assertEqual(response['result'][0]['action'], 'some_action')
        self.assertEqual(response['result'][0]['user'], {'username': 'admin'})
        db.session.delete(log)
        db.session.commit()

    def test_get_list_not_allowed(self):
        if False:
            return 10
        '\n        Log API: Test get list\n        '
        admin_user = self.get_user('admin')
        log = self.insert_log('action', admin_user)
        self.login(username='gamma')
        uri = 'api/v1/log/'
        rv = self.client.get(uri)
        self.assertEqual(rv.status_code, 403)
        self.login(username='alpha')
        rv = self.client.get(uri)
        self.assertEqual(rv.status_code, 403)
        db.session.delete(log)
        db.session.commit()

    def test_get_item(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Log API: Test get item\n        '
        admin_user = self.get_user('admin')
        log = self.insert_log('some_action', admin_user)
        self.login(username='admin')
        uri = f'api/v1/log/{log.id}'
        rv = self.client.get(uri)
        self.assertEqual(rv.status_code, 200)
        response = json.loads(rv.data.decode('utf-8'))
        self.assertEqual(list(response['result'].keys()), EXPECTED_COLUMNS)
        self.assertEqual(response['result']['action'], 'some_action')
        self.assertEqual(response['result']['user'], {'username': 'admin'})
        db.session.delete(log)
        db.session.commit()

    def test_delete_log(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Log API: Test delete (does not exist)\n        '
        admin_user = self.get_user('admin')
        log = self.insert_log('action', admin_user)
        self.login(username='admin')
        uri = f'api/v1/log/{log.id}'
        rv = self.client.delete(uri)
        self.assertEqual(rv.status_code, 405)
        db.session.delete(log)
        db.session.commit()

    def test_update_log(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Log API: Test update (does not exist)\n        '
        admin_user = self.get_user('admin')
        log = self.insert_log('action', admin_user)
        self.login(username='admin')
        log_data = {'action': 'some_action'}
        uri = f'api/v1/log/{log.id}'
        rv = self.client.put(uri, json=log_data)
        self.assertEqual(rv.status_code, 405)
        db.session.delete(log)
        db.session.commit()

    def test_get_recent_activity(self):
        if False:
            i = 10
            return i + 15
        '\n        Log API: Test recent activity endpoint\n        '
        admin_user = self.get_user('admin')
        self.login(username='admin')
        dash = create_dashboard('dash_slug', 'dash_title', '{}', [])
        log1 = self.insert_log('dashboard', admin_user, dashboard_id=dash.id)
        log2 = self.insert_log('dashboard', admin_user, dashboard_id=dash.id)
        uri = f'api/v1/log/recent_activity/'
        rv = self.client.get(uri)
        self.assertEqual(rv.status_code, 200)
        response = json.loads(rv.data.decode('utf-8'))
        db.session.delete(log1)
        db.session.delete(log2)
        db.session.delete(dash)
        db.session.commit()
        self.assertEqual(response, {'result': [{'action': 'dashboard', 'item_type': 'dashboard', 'item_url': '/superset/dashboard/dash_slug/', 'item_title': 'dash_title', 'time': ANY, 'time_delta_humanized': ANY}]})

    def test_get_recent_activity_actions_filter(self):
        if False:
            return 10
        '\n        Log API: Test recent activity actions argument\n        '
        admin_user = self.get_user('admin')
        self.login(username='admin')
        dash = create_dashboard('dash_slug', 'dash_title', '{}', [])
        log = self.insert_log('dashboard', admin_user, dashboard_id=dash.id)
        log2 = self.insert_log('explore', admin_user, dashboard_id=dash.id)
        arguments = {'actions': ['dashboard']}
        uri = f'api/v1/log/recent_activity/?q={prison.dumps(arguments)}'
        rv = self.client.get(uri)
        db.session.delete(log)
        db.session.delete(log2)
        db.session.delete(dash)
        db.session.commit()
        self.assertEqual(rv.status_code, 200)
        response = json.loads(rv.data.decode('utf-8'))
        self.assertEqual(len(response['result']), 1)

    def test_get_recent_activity_distinct_false(self):
        if False:
            return 10
        '\n        Log API: Test recent activity when distinct is false\n        '
        db.session.query(Log).delete(synchronize_session=False)
        db.session.commit()
        admin_user = self.get_user('admin')
        self.login(username='admin')
        dash = create_dashboard('dash_slug', 'dash_title', '{}', [])
        log = self.insert_log('dashboard', admin_user, dashboard_id=dash.id)
        log2 = self.insert_log('dashboard', admin_user, dashboard_id=dash.id)
        arguments = {'distinct': False}
        uri = f'api/v1/log/recent_activity/?q={prison.dumps(arguments)}'
        rv = self.client.get(uri)
        db.session.delete(log)
        db.session.delete(log2)
        db.session.delete(dash)
        db.session.commit()
        self.assertEqual(rv.status_code, 200)
        response = json.loads(rv.data.decode('utf-8'))
        self.assertEqual(len(response['result']), 2)

    def test_get_recent_activity_pagination(self):
        if False:
            while True:
                i = 10
        '\n        Log API: Test recent activity pagination arguments\n        '
        admin_user = self.get_user('admin')
        self.login(username='admin')
        dash = create_dashboard('dash_slug', 'dash_title', '{}', [])
        dash2 = create_dashboard('dash2_slug', 'dash2_title', '{}', [])
        dash3 = create_dashboard('dash3_slug', 'dash3_title', '{}', [])
        log = self.insert_log('dashboard', admin_user, dashboard_id=dash.id)
        log2 = self.insert_log('dashboard', admin_user, dashboard_id=dash2.id)
        log3 = self.insert_log('dashboard', admin_user, dashboard_id=dash3.id)
        now = datetime.now()
        log3.dttm = now
        log2.dttm = now - timedelta(days=1)
        log.dttm = now - timedelta(days=2)
        arguments = {'page': 0, 'page_size': 2}
        uri = f'api/v1/log/recent_activity/?q={prison.dumps(arguments)}'
        rv = self.client.get(uri)
        self.assertEqual(rv.status_code, 200)
        response = json.loads(rv.data.decode('utf-8'))
        self.assertEqual(response, {'result': [{'action': 'dashboard', 'item_type': 'dashboard', 'item_url': '/superset/dashboard/dash3_slug/', 'item_title': 'dash3_title', 'time': ANY, 'time_delta_humanized': ANY}, {'action': 'dashboard', 'item_type': 'dashboard', 'item_url': '/superset/dashboard/dash2_slug/', 'item_title': 'dash2_title', 'time': ANY, 'time_delta_humanized': ANY}]})
        arguments = {'page': 1, 'page_size': 2}
        uri = f'api/v1/log/recent_activity/?q={prison.dumps(arguments)}'
        rv = self.client.get(uri)
        db.session.delete(log)
        db.session.delete(log2)
        db.session.delete(log3)
        db.session.delete(dash)
        db.session.delete(dash2)
        db.session.delete(dash3)
        db.session.commit()
        self.assertEqual(rv.status_code, 200)
        response = json.loads(rv.data.decode('utf-8'))
        self.assertEqual(response, {'result': [{'action': 'dashboard', 'item_type': 'dashboard', 'item_url': '/superset/dashboard/dash_slug/', 'item_title': 'dash_title', 'time': ANY, 'time_delta_humanized': ANY}]})