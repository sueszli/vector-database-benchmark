import prison
import pytest
from superset import db
from superset.connectors.sqla.models import SqlaTable
from superset.models.dashboard import Dashboard
from superset.models.slice import Slice
from tests.integration_tests.fixtures.birth_names_dashboard import load_birth_names_dashboard_with_slices, load_birth_names_data
from tests.integration_tests.fixtures.public_role import public_role_like_gamma
from tests.integration_tests.insert_chart_mixin import InsertChartMixin
from .base_tests import SupersetTestCase

class TestProfile(SupersetTestCase, InsertChartMixin):

    def insert_dashboard_created_by(self, username: str) -> Dashboard:
        if False:
            print('Hello World!')
        user = self.get_user(username)
        dashboard = self.insert_dashboard(f'create_title_test', f'create_slug_test', [user.id], created_by=user)
        return dashboard

    @pytest.fixture()
    def insert_dashboard_created_by_admin(self):
        if False:
            print('Hello World!')
        with self.create_app().app_context():
            dashboard = self.insert_dashboard_created_by('admin')
            yield dashboard
            db.session.delete(dashboard)
            db.session.commit()

    def insert_chart_created_by(self, username: str) -> Slice:
        if False:
            i = 10
            return i + 15
        user = self.get_user(username)
        dataset = db.session.query(SqlaTable).first()
        chart = self.insert_chart(f'create_title_test', [user.id], dataset.id, created_by=user)
        return chart

    @pytest.fixture()
    def insert_chart_created_by_admin(self):
        if False:
            print('Hello World!')
        with self.create_app().app_context():
            chart = self.insert_chart_created_by('admin')
            yield chart
            db.session.delete(chart)
            db.session.commit()

    @pytest.mark.usefixtures('insert_dashboard_created_by_admin')
    @pytest.mark.usefixtures('insert_chart_created_by_admin')
    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_user_profile(self, username='admin'):
        if False:
            for i in range(10):
                print('nop')
        self.login(username=username)
        slc = self.get_slice('Girls', db.session)
        dashboard = db.session.query(Dashboard).filter_by(slug='births').first()
        self.client.post(f'/api/v1/dashboard/{dashboard.id}/favorites/', json={})
        self.client.post(f'/api/v1/chart/{slc.id}/favorites/', json={})
        request_query = {'columns': ['created_on_delta_humanized', 'dashboard_title', 'url'], 'filters': [{'col': 'id', 'opr': 'dashboard_is_favorite', 'value': True}], 'keys': ['none'], 'order_column': 'changed_on', 'order_direction': 'desc', 'page': 0, 'page_size': 100}
        url = f'/api/v1/dashboard/?q={prison.dumps(request_query)}'
        resp = self.client.get(url)
        assert resp.json['count'] == 1
        assert resp.json['result'][0]['dashboard_title'] == 'USA Births Names'
        request_query = {'filters': [{'col': 'id', 'opr': 'chart_is_favorite', 'value': True}], 'order_column': 'slice_name', 'order_direction': 'asc', 'page': 0, 'page_size': 25}
        url = f'api/v1/chart/?q={prison.dumps(request_query)}'
        resp = self.client.get(url)
        assert resp.json['count'] == 1
        assert resp.json['result'][0]['id'] == slc.id
        url = '/api/v1/log/recent_activity/?q=(page_size:50)'
        resp = self.client.get(url)
        assert resp.status_code == 200
        request_query = {'columns': ['created_on_delta_humanized', 'dashboard_title', 'url'], 'filters': [{'col': 'created_by', 'opr': 'dashboard_created_by_me', 'value': 'me'}], 'keys': ['none'], 'order_column': 'changed_on', 'order_direction': 'desc', 'page': 0, 'page_size': 100}
        url = f'/api/v1/dashboard/?q={prison.dumps(request_query)}'
        resp = self.client.get(url)
        assert resp.json['result'][0]['dashboard_title'] == 'create_title_test'
        request_query = {'columns': ['created_on_delta_humanized', 'slice_name', 'url'], 'filters': [{'col': 'created_by', 'opr': 'chart_created_by_me', 'value': 'me'}], 'keys': ['none'], 'order_column': 'changed_on_delta_humanized', 'order_direction': 'desc', 'page': 0, 'page_size': 100}
        url = f'/api/v1/chart/?q={prison.dumps(request_query)}'
        resp = self.client.get(url)
        assert resp.json['count'] == 1
        assert resp.json['result'][0]['slice_name'] == 'create_title_test'
        resp = self.get_resp(f'/profile/')
        self.assertIn('"app"', resp)

    def test_user_profile_gamma(self):
        if False:
            print('Hello World!')
        self.login(username='gamma')
        resp = self.get_resp(f'/profile/')
        self.assertIn('"app"', resp)

    @pytest.mark.usefixtures('public_role_like_gamma')
    def test_user_profile_anonymous(self):
        if False:
            print('Hello World!')
        self.logout()
        resp = self.client.get('/profile/')
        assert resp.status_code == 404