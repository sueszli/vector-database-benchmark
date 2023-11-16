"""Unit tests for Superset"""
import re
import unittest
from random import random
import pytest
from flask import Response, escape, url_for
from sqlalchemy import func
from tests.integration_tests.test_app import app
from superset import db, security_manager
from superset.connectors.sqla.models import SqlaTable
from superset.models.dashboard import Dashboard
from superset.models.slice import Slice
from tests.integration_tests.fixtures.birth_names_dashboard import load_birth_names_dashboard_with_slices, load_birth_names_data
from tests.integration_tests.fixtures.energy_dashboard import load_energy_table_with_slice, load_energy_table_data
from tests.integration_tests.fixtures.public_role import public_role_like_gamma
from tests.integration_tests.fixtures.unicode_dashboard import load_unicode_dashboard_with_position, load_unicode_data
from tests.integration_tests.fixtures.world_bank_dashboard import load_world_bank_dashboard_with_slices, load_world_bank_data
from .base_tests import SupersetTestCase

class TestDashboard(SupersetTestCase):

    @pytest.fixture
    def load_dashboard(self):
        if False:
            return 10
        with app.app_context():
            table = db.session.query(SqlaTable).filter_by(table_name='energy_usage').one()
            slice = db.session.query(Slice).filter_by(slice_name='Energy Sankey').one()
            self.grant_public_access_to_table(table)
            pytest.hidden_dash_slug = f'hidden_dash_{random()}'
            pytest.published_dash_slug = f'published_dash_{random()}'
            published_dash = Dashboard()
            published_dash.dashboard_title = 'Published Dashboard'
            published_dash.slug = pytest.published_dash_slug
            published_dash.slices = [slice]
            published_dash.published = True
            hidden_dash = Dashboard()
            hidden_dash.dashboard_title = 'Hidden Dashboard'
            hidden_dash.slug = pytest.hidden_dash_slug
            hidden_dash.slices = [slice]
            hidden_dash.published = False
            db.session.merge(published_dash)
            db.session.merge(hidden_dash)
            yield db.session.commit()
            self.revoke_public_access_to_table(table)
            db.session.delete(published_dash)
            db.session.delete(hidden_dash)
            db.session.commit()

    def get_mock_positions(self, dash):
        if False:
            print('Hello World!')
        positions = {'DASHBOARD_VERSION_KEY': 'v2'}
        for (i, slc) in enumerate(dash.slices):
            id = f'DASHBOARD_CHART_TYPE-{i}'
            d = {'type': 'CHART', 'id': id, 'children': [], 'meta': {'width': 4, 'height': 50, 'chartId': slc.id}}
            positions[id] = d
        return positions

    def test_get_dashboard(self):
        if False:
            while True:
                i = 10
        self.login(username='admin')
        for dash in db.session.query(Dashboard):
            assert escape(dash.dashboard_title) in self.client.get(dash.url).get_data(as_text=True)

    def test_superset_dashboard_url(self):
        if False:
            for i in range(10):
                print('nop')
        url_for('Superset.dashboard', dashboard_id_or_slug=1)

    def test_new_dashboard(self):
        if False:
            for i in range(10):
                print('nop')
        self.login(username='admin')
        dash_count_before = db.session.query(func.count(Dashboard.id)).first()[0]
        url = '/dashboard/new/'
        response = self.client.get(url, follow_redirects=False)
        dash_count_after = db.session.query(func.count(Dashboard.id)).first()[0]
        self.assertEqual(dash_count_before + 1, dash_count_after)
        group = re.match('\\/superset\\/dashboard\\/([0-9]*)\\/\\?edit=true', response.headers['Location'])
        assert group is not None
        created_dashboard_id = int(group[1])
        created_dashboard = db.session.query(Dashboard).get(created_dashboard_id)
        db.session.delete(created_dashboard)
        db.session.commit()

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    @pytest.mark.usefixtures('public_role_like_gamma')
    def test_public_user_dashboard_access(self):
        if False:
            return 10
        table = db.session.query(SqlaTable).filter_by(table_name='birth_names').one()
        births_dash = db.session.query(Dashboard).filter_by(slug='births').one()
        births_dash.published = True
        db.session.merge(births_dash)
        db.session.commit()
        self.revoke_public_access_to_table(table)
        self.logout()
        resp = self.get_resp('/api/v1/chart/')
        self.assertNotIn('birth_names', resp)
        resp = self.get_resp('/api/v1/dashboard/')
        self.assertNotIn('/superset/dashboard/births/', resp)
        self.grant_public_access_to_table(table)
        self.assertIn('birth_names', self.get_resp('/api/v1/chart/'))
        resp = self.get_resp('/api/v1/dashboard/')
        self.assertIn('/superset/dashboard/births/', resp)
        resp = self.get_resp('/api/v1/chart/')
        self.assertNotIn('wb_health_population', resp)
        resp = self.get_resp('/api/v1/dashboard/')
        self.assertNotIn('/superset/dashboard/world_health/', resp)
        self.revoke_public_access_to_table(table)

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices', 'public_role_like_gamma')
    def test_dashboard_with_created_by_can_be_accessed_by_public_users(self):
        if False:
            print('Hello World!')
        self.logout()
        table = db.session.query(SqlaTable).filter_by(table_name='birth_names').one()
        self.grant_public_access_to_table(table)
        dash = db.session.query(Dashboard).filter_by(slug='births').first()
        dash.owners = [security_manager.find_user('admin')]
        dash.created_by = security_manager.find_user('admin')
        db.session.merge(dash)
        db.session.commit()
        res: Response = self.client.get('/superset/dashboard/births/')
        assert res.status_code == 200
        self.revoke_public_access_to_table(table)

    @pytest.mark.usefixtures('load_energy_table_with_slice', 'load_dashboard')
    def test_users_can_list_published_dashboard(self):
        if False:
            print('Hello World!')
        self.login('alpha')
        resp = self.get_resp('/api/v1/dashboard/')
        assert f'/superset/dashboard/{pytest.hidden_dash_slug}/' not in resp
        assert f'/superset/dashboard/{pytest.published_dash_slug}/' in resp

    def test_users_can_view_own_dashboard(self):
        if False:
            for i in range(10):
                print('nop')
        user = security_manager.find_user('gamma')
        my_dash_slug = f'my_dash_{random()}'
        not_my_dash_slug = f'not_my_dash_{random()}'
        dash = Dashboard()
        dash.dashboard_title = 'My Dashboard'
        dash.slug = my_dash_slug
        dash.owners = [user]
        hidden_dash = Dashboard()
        hidden_dash.dashboard_title = 'Not My Dashboard'
        hidden_dash.slug = not_my_dash_slug
        db.session.add(dash)
        db.session.add(hidden_dash)
        db.session.commit()
        self.login(user.username)
        resp = self.get_resp('/api/v1/dashboard/')
        db.session.delete(dash)
        db.session.delete(hidden_dash)
        db.session.commit()
        self.assertIn(f'/superset/dashboard/{my_dash_slug}/', resp)
        self.assertNotIn(f'/superset/dashboard/{not_my_dash_slug}/', resp)

    def test_user_can_not_view_unpublished_dash(self):
        if False:
            while True:
                i = 10
        admin_user = security_manager.find_user('admin')
        gamma_user = security_manager.find_user('gamma')
        slug = f'admin_owned_unpublished_dash_{random()}'
        dash = Dashboard()
        dash.dashboard_title = 'My Dashboard'
        dash.slug = slug
        dash.owners = [admin_user]
        dash.published = False
        db.session.add(dash)
        db.session.commit()
        self.login(gamma_user.username)
        resp = self.get_resp('/api/v1/dashboard/')
        db.session.delete(dash)
        db.session.commit()
        self.assertNotIn(f'/superset/dashboard/{slug}/', resp)
if __name__ == '__main__':
    unittest.main()