from __future__ import annotations
from typing import Any, TYPE_CHECKING
from tests.integration_tests.dashboards.filter_sets.consts import DASHBOARD_OWNER_USERNAME, FILTER_SET_OWNER_USERNAME, REGULAR_USER
from tests.integration_tests.dashboards.filter_sets.utils import call_delete_filter_set, collect_all_ids, get_filter_set_by_name
from tests.integration_tests.test_app import login
if TYPE_CHECKING:
    from flask.testing import FlaskClient
    from superset.models.filter_set import FilterSet

def assert_filterset_was_not_deleted(filter_set_dict: dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    assert get_filter_set_by_name(filter_set_dict['name']) is not None

def assert_filterset_deleted(filter_set_dict: dict[str, Any]) -> None:
    if False:
        while True:
            i = 10
    assert get_filter_set_by_name(filter_set_dict['name']) is None

class TestDeleteFilterSet:

    def test_with_dashboard_exists_filterset_not_exists__200(self, dashboard_id: int, filtersets: dict[str, list[FilterSet]], client: FlaskClient[Any]):
        if False:
            print('Hello World!')
        login(client, 'admin')
        filter_set_id = max(collect_all_ids(filtersets)) + 1
        response = call_delete_filter_set(client, {'id': filter_set_id}, dashboard_id)
        assert response.status_code == 200

    def test_with_dashboard_not_exists_filterset_not_exists__404(self, not_exists_dashboard_id: int, filtersets: dict[str, list[FilterSet]], client: FlaskClient[Any]):
        if False:
            i = 10
            return i + 15
        login(client, 'admin')
        filter_set_id = max(collect_all_ids(filtersets)) + 1
        response = call_delete_filter_set(client, {'id': filter_set_id}, not_exists_dashboard_id)
        assert response.status_code == 404

    def test_with_dashboard_not_exists_filterset_exists__404(self, not_exists_dashboard_id: int, dashboard_based_filter_set_dict: dict[str, Any], client: FlaskClient[Any]):
        if False:
            print('Hello World!')
        login(client, 'admin')
        response = call_delete_filter_set(client, dashboard_based_filter_set_dict, not_exists_dashboard_id)
        assert response.status_code == 404
        assert_filterset_was_not_deleted(dashboard_based_filter_set_dict)

    def test_when_caller_is_admin_and_owner_type_is_user__200(self, test_users: dict[str, int], user_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            for i in range(10):
                print('nop')
        login(client, 'admin')
        response = call_delete_filter_set(client, user_based_filter_set_dict)
        assert response.status_code == 200
        assert_filterset_deleted(user_based_filter_set_dict)

    def test_when_caller_is_admin_and_owner_type_is_dashboard__200(self, test_users: dict[str, int], dashboard_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            i = 10
            return i + 15
        login(client, 'admin')
        response = call_delete_filter_set(client, dashboard_based_filter_set_dict)
        assert response.status_code == 200
        assert_filterset_deleted(dashboard_based_filter_set_dict)

    def test_when_caller_is_dashboard_owner_and_owner_is_other_user_403(self, test_users: dict[str, int], user_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            return 10
        login(client, DASHBOARD_OWNER_USERNAME)
        response = call_delete_filter_set(client, user_based_filter_set_dict)
        assert response.status_code == 403
        assert_filterset_was_not_deleted(user_based_filter_set_dict)

    def test_when_caller_is_dashboard_owner_and_owner_type_is_dashboard__200(self, test_users: dict[str, int], dashboard_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            print('Hello World!')
        login(client, DASHBOARD_OWNER_USERNAME)
        response = call_delete_filter_set(client, dashboard_based_filter_set_dict)
        assert response.status_code == 200
        assert_filterset_deleted(dashboard_based_filter_set_dict)

    def test_when_caller_is_filterset_owner__200(self, test_users: dict[str, int], user_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            return 10
        login(client, FILTER_SET_OWNER_USERNAME)
        response = call_delete_filter_set(client, user_based_filter_set_dict)
        assert response.status_code == 200
        assert_filterset_deleted(user_based_filter_set_dict)

    def test_when_caller_is_regular_user_and_owner_type_is_user__403(self, test_users: dict[str, int], user_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            for i in range(10):
                print('nop')
        login(client, REGULAR_USER)
        response = call_delete_filter_set(client, user_based_filter_set_dict)
        assert response.status_code == 403
        assert_filterset_was_not_deleted(user_based_filter_set_dict)

    def test_when_caller_is_regular_user_and_owner_type_is_dashboard__403(self, test_users: dict[str, int], dashboard_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            print('Hello World!')
        login(client, REGULAR_USER)
        response = call_delete_filter_set(client, dashboard_based_filter_set_dict)
        assert response.status_code == 403
        assert_filterset_was_not_deleted(dashboard_based_filter_set_dict)