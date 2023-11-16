"""Unit tests for Superset"""
import json
from unittest import mock
from unittest.mock import patch
import pytest
from superset.daos.dashboard import DashboardDAO
from superset.dashboards.commands.exceptions import DashboardForbiddenError
from superset.utils.core import backend, override_user
from tests.integration_tests.conftest import with_feature_flags
from tests.integration_tests.dashboards.dashboard_test_utils import *
from tests.integration_tests.dashboards.security.base_case import BaseTestDashboardSecurity
from tests.integration_tests.dashboards.superset_factory_util import create_dashboard_to_db, create_database_to_db, create_datasource_table_to_db, create_slice_to_db
from tests.integration_tests.fixtures.birth_names_dashboard import load_birth_names_dashboard_with_slices, load_birth_names_data
from tests.integration_tests.fixtures.public_role import public_role_like_gamma
from tests.integration_tests.fixtures.query_context import get_query_context
from tests.integration_tests.fixtures.world_bank_dashboard import load_world_bank_dashboard_with_slices, load_world_bank_data
CHART_DATA_URI = 'api/v1/chart/data'

@mock.patch.dict('superset.extensions.feature_flag_manager._feature_flags', DASHBOARD_RBAC=True)
class TestDashboardRoleBasedSecurity(BaseTestDashboardSecurity):

    def test_get_dashboard_view__admin_can_access(self):
        if False:
            return 10
        dashboard_to_access = create_dashboard_to_db(owners=[], slices=[create_slice_to_db()], published=False)
        self.login('admin')
        response = self.get_dashboard_view_response(dashboard_to_access)
        self.assert200(response)

    def test_get_dashboard_view__owner_can_access(self):
        if False:
            for i in range(10):
                print('nop')
        username = random_str()
        new_role = f'role_{random_str()}'
        owner = self.create_user_with_roles(username, [new_role], should_create_roles=True)
        dashboard_to_access = create_dashboard_to_db(owners=[owner], slices=[create_slice_to_db()], published=False)
        self.login(username)
        response = self.get_dashboard_view_response(dashboard_to_access)
        self.assert200(response)

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_get_dashboard_view__user_can_not_access_without_permission(self):
        if False:
            for i in range(10):
                print('nop')
        username = random_str()
        new_role = f'role_{random_str()}'
        self.create_user_with_roles(username, [new_role], should_create_roles=True)
        slice = db.session.query(Slice).filter_by(slice_name='Girl Name Cloud').one_or_none()
        dashboard_to_access = create_dashboard_to_db(published=True, slices=[slice])
        self.login(username)
        response = self.get_dashboard_view_response(dashboard_to_access)
        assert response.status_code == 302
        request_payload = get_query_context('birth_names')
        rv = self.post_assert_metric(CHART_DATA_URI, request_payload, 'data')
        assert rv.status_code == 403

    def test_get_dashboard_view__user_with_dashboard_permission_can_not_access_draft(self):
        if False:
            return 10
        dashboard_to_access = create_dashboard_to_db(published=False)
        username = random_str()
        new_role = f'role_{random_str()}'
        self.create_user_with_roles(username, [new_role], should_create_roles=True)
        grant_access_to_dashboard(dashboard_to_access, new_role)
        self.login(username)
        response = self.get_dashboard_view_response(dashboard_to_access)
        assert response.status_code == 302
        revoke_access_to_dashboard(dashboard_to_access, new_role)

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_get_dashboard_view__user_no_access_regular_rbac(self):
        if False:
            return 10
        if backend() == 'hive':
            return
        slice = db.session.query(Slice).filter_by(slice_name='Girl Name Cloud').one_or_none()
        dashboard = create_dashboard_to_db(published=True, slices=[slice])
        self.login('gamma')
        response = self.get_dashboard_view_response(dashboard)
        assert response.status_code == 302
        request_payload = get_query_context('birth_names')
        rv = self.post_assert_metric(CHART_DATA_URI, request_payload, 'data')
        assert rv.status_code == 403
        db.session.delete(dashboard)
        db.session.commit()

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_get_dashboard_view__user_access_regular_rbac(self):
        if False:
            return 10
        if backend() == 'hive':
            return
        slice = db.session.query(Slice).filter_by(slice_name='Girl Name Cloud').one_or_none()
        dashboard = create_dashboard_to_db(published=True, slices=[slice])
        self.login('gamma_sqllab')
        response = self.get_dashboard_view_response(dashboard)
        assert response.status_code == 200
        request_payload = get_query_context('birth_names')
        rv = self.post_assert_metric(CHART_DATA_URI, request_payload, 'data')
        assert rv.status_code == 200
        db.session.delete(dashboard)
        db.session.commit()

    @pytest.mark.usefixtures('load_birth_names_dashboard_with_slices')
    def test_get_dashboard_view__user_access_with_dashboard_permission(self):
        if False:
            print('Hello World!')
        if backend() == 'hive':
            return
        username = random_str()
        new_role = f'role_{random_str()}'
        self.create_user_with_roles(username, [new_role], should_create_roles=True)
        slice = db.session.query(Slice).filter_by(slice_name='Girl Name Cloud').one_or_none()
        dashboard_to_access = create_dashboard_to_db(published=True, slices=[slice])
        self.login(username)
        grant_access_to_dashboard(dashboard_to_access, new_role)
        response = self.get_dashboard_view_response(dashboard_to_access)
        self.assert200(response)
        request_payload = get_query_context('birth_names')
        rv = self.post_assert_metric(CHART_DATA_URI, request_payload, 'data')
        self.assertEqual(rv.status_code, 403)
        revoke_access_to_dashboard(dashboard_to_access, new_role)

    @pytest.mark.usefixtures('public_role_like_gamma')
    def test_get_dashboard_view__public_user_can_not_access_without_permission(self):
        if False:
            while True:
                i = 10
        dashboard_to_access = create_dashboard_to_db(published=True)
        grant_access_to_dashboard(dashboard_to_access, 'Alpha')
        self.logout()
        response = self.get_dashboard_view_response(dashboard_to_access)
        assert response.status_code == 302

    @pytest.mark.usefixtures('public_role_like_gamma')
    def test_get_dashboard_view__public_user_with_dashboard_permission_can_not_access_draft(self):
        if False:
            print('Hello World!')
        dashboard_to_access = create_dashboard_to_db(published=False)
        grant_access_to_dashboard(dashboard_to_access, 'Public')
        self.logout()
        response = self.get_dashboard_view_response(dashboard_to_access)
        assert response.status_code == 302
        revoke_access_to_dashboard(dashboard_to_access, 'Public')

    @pytest.mark.usefixtures('public_role_like_gamma')
    def test_get_dashboard_view__public_user_access_with_dashboard_permission(self):
        if False:
            for i in range(10):
                print('nop')
        dashboard_to_access = create_dashboard_to_db(published=True, slices=[create_slice_to_db()])
        grant_access_to_dashboard(dashboard_to_access, 'Public')
        self.logout()
        response = self.get_dashboard_view_response(dashboard_to_access)
        self.assert200(response)
        revoke_access_to_dashboard(dashboard_to_access, 'Public')

    def _create_sample_dashboards_with_owner_access(self):
        if False:
            return 10
        username = random_str()
        new_role = f'role_{random_str()}'
        owner = self.create_user_with_roles(username, [new_role], should_create_roles=True)
        database = create_database_to_db()
        table = create_datasource_table_to_db(db_id=database.id, owners=[owner])
        first_dash = create_dashboard_to_db(owners=[owner], slices=[create_slice_to_db(datasource_id=table.id)])
        second_dash = create_dashboard_to_db(owners=[owner], slices=[create_slice_to_db(datasource_id=table.id)])
        owned_dashboards = [first_dash, second_dash]
        not_owned_dashboards = [create_dashboard_to_db(slices=[create_slice_to_db(datasource_id=table.id)], published=True)]
        self.login(username)
        return (not_owned_dashboards, owned_dashboards)

    def _create_sample_only_published_dashboard_with_roles(self):
        if False:
            i = 10
            return i + 15
        username = random_str()
        new_role = f'role_{random_str()}'
        self.create_user_with_roles(username, [new_role], should_create_roles=True)
        published_dashboards = [create_dashboard_to_db(published=True), create_dashboard_to_db(published=True)]
        draft_dashboards = [create_dashboard_to_db(published=False), create_dashboard_to_db(published=False)]
        for dash in published_dashboards + draft_dashboards:
            grant_access_to_dashboard(dash, new_role)
        self.login(username)
        return (new_role, draft_dashboards, published_dashboards)

    def test_get_dashboards_api__admin_get_all_dashboards(self):
        if False:
            while True:
                i = 10
        create_dashboard_to_db(owners=[], slices=[create_slice_to_db()], published=False)
        dashboard_counts = count_dashboards()
        self.login('admin')
        response = self.get_dashboards_api_response()
        self.assert_dashboards_api_response(response, dashboard_counts)

    def test_get_dashboards_api__owner_get_all_owned_dashboards(self):
        if False:
            return 10
        (not_owned_dashboards, owned_dashboards) = self._create_sample_dashboards_with_owner_access()
        response = self.get_dashboards_api_response()
        self.assert_dashboards_api_response(response, 2, owned_dashboards, not_owned_dashboards)

    def test_get_dashboards_api__user_without_any_permissions_get_empty_list(self):
        if False:
            return 10
        username = random_str()
        new_role = f'role_{random_str()}'
        self.create_user_with_roles(username, [new_role], should_create_roles=True)
        create_dashboard_to_db(published=True)
        self.login(username)
        response = self.get_dashboards_api_response()
        self.assert_dashboards_api_response(response, 0)

    def test_get_dashboards_api__user_get_only_published_permitted_dashboards(self):
        if False:
            return 10
        (new_role, draft_dashboards, published_dashboards) = self._create_sample_only_published_dashboard_with_roles()
        response = self.get_dashboards_api_response()
        self.assert_dashboards_api_response(response, len(published_dashboards), published_dashboards, draft_dashboards)
        for dash in published_dashboards + draft_dashboards:
            revoke_access_to_dashboard(dash, new_role)

    @pytest.mark.usefixtures('public_role_like_gamma')
    def test_get_dashboards_api__public_user_without_any_permissions_get_empty_list(self):
        if False:
            for i in range(10):
                print('nop')
        create_dashboard_to_db(published=True)
        self.logout()
        response = self.get_dashboards_api_response()
        self.assert_dashboards_api_response(response, 0)

    @pytest.mark.usefixtures('public_role_like_gamma')
    def test_get_dashboards_api__public_user_get_only_published_permitted_dashboards(self):
        if False:
            i = 10
            return i + 15
        published_dashboards = [create_dashboard_to_db(published=True), create_dashboard_to_db(published=True)]
        draft_dashboards = [create_dashboard_to_db(published=False), create_dashboard_to_db(published=False)]
        for dash in published_dashboards + draft_dashboards:
            grant_access_to_dashboard(dash, 'Public')
        self.logout()
        response = self.get_dashboards_api_response()
        self.assert_dashboards_api_response(response, len(published_dashboards), published_dashboards, draft_dashboards)
        for dash in published_dashboards + draft_dashboards:
            revoke_access_to_dashboard(dash, 'Public')

    def test_get_draft_dashboard_without_roles_by_uuid(self):
        if False:
            print('Hello World!')
        '\n        Dashboard API: Test get draft dashboard without roles by uuid\n        '
        admin = self.get_user('admin')
        dashboard = self.insert_dashboard('title', 'slug1', [admin.id])
        assert not dashboard.published
        assert dashboard.roles == []
        self.login(username='gamma')
        uri = f'api/v1/dashboard/{dashboard.uuid}'
        rv = self.client.get(uri)
        assert rv.status_code == 200
        db.session.delete(dashboard)
        db.session.commit()

    def test_cannot_get_draft_dashboard_with_roles_by_uuid(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Dashboard API: Test get dashboard by uuid\n        '
        admin = self.get_user('admin')
        admin_role = self.get_role('Admin')
        dashboard = self.insert_dashboard('title', 'slug1', [admin.id], roles=[admin_role.id])
        assert not dashboard.published
        assert dashboard.roles == [admin_role]
        self.login(username='gamma')
        uri = f'api/v1/dashboard/{dashboard.uuid}'
        rv = self.client.get(uri)
        assert rv.status_code == 403
        db.session.delete(dashboard)
        db.session.commit()

    @with_feature_flags(DASHBOARD_RBAC=True)
    @pytest.mark.usefixtures('load_world_bank_dashboard_with_slices')
    def test_copy_dashboard_via_api(self):
        if False:
            print('Hello World!')
        source = db.session.query(Dashboard).filter_by(slug='world_health').first()
        source.roles = [self.get_role('Gamma')]
        if not (published := source.published):
            source.published = True
        db.session.commit()
        uri = f'api/v1/dashboard/{source.id}/copy/'
        data = {'dashboard_title': 'copied dash', 'css': '<css>', 'duplicate_slices': False, 'json_metadata': json.dumps({'positions': source.position, 'color_namespace': 'Color Namespace Test', 'color_scheme': 'Color Scheme Test'})}
        self.login(username='gamma')
        rv = self.client.post(uri, json=data)
        self.assertEqual(rv.status_code, 403)
        self.logout()
        self.login(username='admin')
        rv = self.client.post(uri, json=data)
        self.assertEqual(rv.status_code, 200)
        self.logout()
        response = json.loads(rv.data.decode('utf-8'))
        target = db.session.query(Dashboard).filter(Dashboard.id == response['result']['id']).one()
        db.session.delete(target)
        source.roles = []
        if not published:
            source.published = False
        db.session.commit()

    @with_feature_flags(DASHBOARD_RBAC=True)
    @pytest.mark.usefixtures('load_world_bank_dashboard_with_slices')
    def test_copy_dashboard_via_dao(self):
        if False:
            i = 10
            return i + 15
        source = db.session.query(Dashboard).filter_by(slug='world_health').first()
        data = {'dashboard_title': 'copied dash', 'css': '<css>', 'duplicate_slices': False, 'json_metadata': json.dumps({'positions': source.position, 'color_namespace': 'Color Namespace Test', 'color_scheme': 'Color Scheme Test'})}
        with override_user(security_manager.find_user('gamma')):
            with pytest.raises(DashboardForbiddenError):
                DashboardDAO.copy_dashboard(source, data)
        with override_user(security_manager.find_user('admin')):
            target = DashboardDAO.copy_dashboard(source, data)
            db.session.delete(target)
        db.session.commit()