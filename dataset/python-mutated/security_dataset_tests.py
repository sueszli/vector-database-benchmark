"""Unit tests for Superset"""
import json
import prison
import pytest
from flask import escape
from superset import app
from superset.daos.dashboard import DashboardDAO
from tests.integration_tests.dashboards.base_case import DashboardTestCase
from tests.integration_tests.dashboards.consts import *
from tests.integration_tests.dashboards.dashboard_test_utils import *
from tests.integration_tests.dashboards.superset_factory_util import *
from tests.integration_tests.fixtures.energy_dashboard import load_energy_table_data, load_energy_table_with_slice

class TestDashboardDatasetSecurity(DashboardTestCase):

    @pytest.fixture
    def load_dashboard(self):
        if False:
            for i in range(10):
                print('nop')
        with app.app_context():
            table = db.session.query(SqlaTable).filter_by(table_name='energy_usage').one()
            slice = db.session.query(Slice).filter_by(slice_name='Energy Sankey').one()
            self.grant_public_access_to_table(table)
            pytest.hidden_dash_slug = f'hidden_dash_{random_slug()}'
            pytest.published_dash_slug = f'published_dash_{random_slug()}'
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

    def test_dashboard_access__admin_can_access_all(self):
        if False:
            i = 10
            return i + 15
        self.login(username=ADMIN_USERNAME)
        dashboard_title_by_url = {dash.url: dash.dashboard_title for dash in get_all_dashboards()}
        responses_by_url = {url: self.client.get(url) for url in dashboard_title_by_url.keys()}
        for (dashboard_url, get_dashboard_response) in responses_by_url.items():
            self.assert200(get_dashboard_response)

    def test_get_dashboards__users_are_dashboards_owners(self):
        if False:
            i = 10
            return i + 15
        username = 'gamma'
        user = security_manager.find_user(username)
        my_owned_dashboard = create_dashboard_to_db(dashboard_title='My Dashboard', published=False, owners=[user])
        not_my_owned_dashboard = create_dashboard_to_db(dashboard_title='Not My Dashboard', published=False)
        self.login(user.username)
        get_dashboards_response = self.get_resp(DASHBOARDS_API_URL)
        self.assertIn(my_owned_dashboard.url, get_dashboards_response)
        self.assertNotIn(not_my_owned_dashboard.url, get_dashboards_response)

    def test_get_dashboards__owners_can_view_empty_dashboard(self):
        if False:
            i = 10
            return i + 15
        dash = create_dashboard_to_db('Empty Dashboard', slug='empty_dashboard')
        dashboard_url = dash.url
        gamma_user = security_manager.find_user('gamma')
        self.login(gamma_user.username)
        get_dashboards_response = self.get_resp(DASHBOARDS_API_URL)
        self.assertNotIn(dashboard_url, get_dashboards_response)

    def test_get_dashboards__user_can_not_view_unpublished_dash(self):
        if False:
            print('Hello World!')
        admin_user = security_manager.find_user(ADMIN_USERNAME)
        gamma_user = security_manager.find_user(GAMMA_USERNAME)
        admin_and_draft_dashboard = create_dashboard_to_db(dashboard_title='admin_owned_unpublished_dash', owners=[admin_user])
        self.login(gamma_user.username)
        get_dashboards_response_as_gamma = self.get_resp(DASHBOARDS_API_URL)
        self.assertNotIn(admin_and_draft_dashboard.url, get_dashboards_response_as_gamma)

    @pytest.mark.usefixtures('load_energy_table_with_slice', 'load_dashboard')
    def test_get_dashboards__users_can_view_permitted_dashboard(self):
        if False:
            while True:
                i = 10
        username = random_str()
        new_role = f'role_{random_str()}'
        self.create_user_with_roles(username, [new_role], should_create_roles=True)
        accessed_table = get_sql_table_by_name('energy_usage')
        self.grant_role_access_to_table(accessed_table, new_role)
        slice_to_add_to_dashboards = get_slice_by_name('Energy Sankey')
        first_dash = create_dashboard_to_db(dashboard_title='Published Dashboard', published=True, slices=[slice_to_add_to_dashboards])
        second_dash = create_dashboard_to_db(dashboard_title='Hidden Dashboard', published=True, slices=[slice_to_add_to_dashboards])
        try:
            self.login(username)
            get_dashboards_response = self.get_resp(DASHBOARDS_API_URL)
            self.assertIn(second_dash.url, get_dashboards_response)
            self.assertIn(first_dash.url, get_dashboards_response)
        finally:
            self.revoke_public_access_to_table(accessed_table)

    def test_get_dashboards_api_no_data_access(self):
        if False:
            print('Hello World!')
        '\n        Dashboard API: Test get dashboards no data access\n        '
        admin = self.get_user('admin')
        title = f'title{random_str()}'
        dashboard = create_dashboard_to_db(title, 'slug1', owners=[admin])
        self.login(username='gamma')
        arguments = {'filters': [{'col': 'dashboard_title', 'opr': 'sw', 'value': title[0:8]}]}
        uri = DASHBOARDS_API_URL_WITH_QUERY_FORMAT.format(prison.dumps(arguments))
        rv = self.client.get(uri)
        self.assert200(rv)
        data = json.loads(rv.data.decode('utf-8'))
        self.assertEqual(0, data['count'])
        DashboardDAO.delete(dashboard)