from __future__ import annotations
import json
from typing import Any, TYPE_CHECKING
from superset.dashboards.filter_sets.consts import DESCRIPTION_FIELD, JSON_METADATA_FIELD, NAME_FIELD, OWNER_TYPE_FIELD, PARAMS_PROPERTY
from tests.integration_tests.dashboards.filter_sets.consts import DASHBOARD_OWNER_USERNAME, FILTER_SET_OWNER_USERNAME, REGULAR_USER
from tests.integration_tests.dashboards.filter_sets.utils import call_update_filter_set, collect_all_ids, get_filter_set_by_name
from tests.integration_tests.test_app import login
if TYPE_CHECKING:
    from flask.testing import FlaskClient
    from superset.models.filter_set import FilterSet

def merge_two_filter_set_dict(first: dict[Any, Any], second: dict[Any, Any]) -> dict[Any, Any]:
    if False:
        while True:
            i = 10
    for d in [first, second]:
        if JSON_METADATA_FIELD in d:
            if PARAMS_PROPERTY not in d:
                d.setdefault(PARAMS_PROPERTY, json.loads(d[JSON_METADATA_FIELD]))
            d.pop(JSON_METADATA_FIELD)
    return {**first, **second}

def assert_filterset_was_not_updated(filter_set_dict: dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    assert filter_set_dict == get_filter_set_by_name(filter_set_dict['name']).to_dict()

def assert_filterset_updated(filter_set_dict_before: dict[str, Any], data_updated: dict[str, Any]) -> None:
    if False:
        while True:
            i = 10
    expected_data = merge_two_filter_set_dict(filter_set_dict_before, data_updated)
    assert expected_data == get_filter_set_by_name(expected_data['name']).to_dict()

class TestUpdateFilterSet:

    def test_with_dashboard_exists_filterset_not_exists__404(self, dashboard_id: int, filtersets: dict[str, list[FilterSet]], client: FlaskClient[Any]):
        if False:
            return 10
        login(client, 'admin')
        filter_set_id = max(collect_all_ids(filtersets)) + 1
        response = call_update_filter_set(client, {'id': filter_set_id}, {}, dashboard_id)
        assert response.status_code == 404

    def test_with_dashboard_not_exists_filterset_not_exists__404(self, not_exists_dashboard_id: int, filtersets: dict[str, list[FilterSet]], client: FlaskClient[Any]):
        if False:
            print('Hello World!')
        login(client, 'admin')
        filter_set_id = max(collect_all_ids(filtersets)) + 1
        response = call_update_filter_set(client, {'id': filter_set_id}, {}, not_exists_dashboard_id)
        assert response.status_code == 404

    def test_with_dashboard_not_exists_filterset_exists__404(self, not_exists_dashboard_id: int, dashboard_based_filter_set_dict: dict[str, Any], client: FlaskClient[Any]):
        if False:
            for i in range(10):
                print('nop')
        login(client, 'admin')
        response = call_update_filter_set(client, dashboard_based_filter_set_dict, {}, not_exists_dashboard_id)
        assert response.status_code == 404
        assert_filterset_was_not_updated(dashboard_based_filter_set_dict)

    def test_with_extra_field__400(self, dashboard_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            print('Hello World!')
        login(client, 'admin')
        valid_filter_set_data_for_update['extra'] = 'val'
        response = call_update_filter_set(client, dashboard_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 400
        assert response.json['message']['extra'][0] == 'Unknown field.'
        assert_filterset_was_not_updated(dashboard_based_filter_set_dict)

    def test_with_id_field__400(self, dashboard_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            i = 10
            return i + 15
        login(client, 'admin')
        valid_filter_set_data_for_update['id'] = 1
        response = call_update_filter_set(client, dashboard_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 400
        assert response.json['message']['id'][0] == 'Unknown field.'
        assert_filterset_was_not_updated(dashboard_based_filter_set_dict)

    def test_with_none_name__400(self, dashboard_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            return 10
        login(client, 'admin')
        valid_filter_set_data_for_update[NAME_FIELD] = None
        response = call_update_filter_set(client, dashboard_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 400
        assert_filterset_was_not_updated(dashboard_based_filter_set_dict)

    def test_with_int_as_name__400(self, dashboard_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            print('Hello World!')
        login(client, 'admin')
        valid_filter_set_data_for_update[NAME_FIELD] = 4
        response = call_update_filter_set(client, dashboard_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 400
        assert_filterset_was_not_updated(dashboard_based_filter_set_dict)

    def test_without_name__200(self, dashboard_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            print('Hello World!')
        login(client, 'admin')
        valid_filter_set_data_for_update.pop(NAME_FIELD, None)
        response = call_update_filter_set(client, dashboard_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 200
        assert_filterset_updated(dashboard_based_filter_set_dict, valid_filter_set_data_for_update)

    def test_with_none_description__400(self, dashboard_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            return 10
        login(client, 'admin')
        valid_filter_set_data_for_update[DESCRIPTION_FIELD] = None
        response = call_update_filter_set(client, dashboard_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 400
        assert_filterset_was_not_updated(dashboard_based_filter_set_dict)

    def test_with_int_as_description__400(self, dashboard_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            return 10
        login(client, 'admin')
        valid_filter_set_data_for_update[DESCRIPTION_FIELD] = 1
        response = call_update_filter_set(client, dashboard_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 400
        assert_filterset_was_not_updated(dashboard_based_filter_set_dict)

    def test_without_description__200(self, dashboard_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            return 10
        login(client, 'admin')
        valid_filter_set_data_for_update.pop(DESCRIPTION_FIELD, None)
        response = call_update_filter_set(client, dashboard_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 200
        assert_filterset_updated(dashboard_based_filter_set_dict, valid_filter_set_data_for_update)

    def test_with_invalid_json_metadata__400(self, dashboard_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            while True:
                i = 10
        login(client, 'admin')
        valid_filter_set_data_for_update[DESCRIPTION_FIELD] = {}
        response = call_update_filter_set(client, dashboard_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 400
        assert_filterset_was_not_updated(dashboard_based_filter_set_dict)

    def test_with_json_metadata__200(self, dashboard_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], valid_json_metadata: dict[Any, Any], client: FlaskClient[Any]):
        if False:
            i = 10
            return i + 15
        login(client, 'admin')
        valid_json_metadata['nativeFilters'] = {'changed': 'changed'}
        valid_filter_set_data_for_update[JSON_METADATA_FIELD] = json.dumps(valid_json_metadata)
        response = call_update_filter_set(client, dashboard_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 200
        assert_filterset_updated(dashboard_based_filter_set_dict, valid_filter_set_data_for_update)

    def test_with_invalid_owner_type__400(self, dashboard_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            i = 10
            return i + 15
        login(client, 'admin')
        valid_filter_set_data_for_update[OWNER_TYPE_FIELD] = 'OTHER_TYPE'
        response = call_update_filter_set(client, dashboard_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 400
        assert_filterset_was_not_updated(dashboard_based_filter_set_dict)

    def test_with_user_owner_type__400(self, dashboard_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            return 10
        login(client, 'admin')
        valid_filter_set_data_for_update[OWNER_TYPE_FIELD] = 'User'
        response = call_update_filter_set(client, dashboard_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 400
        assert_filterset_was_not_updated(dashboard_based_filter_set_dict)

    def test_with_dashboard_owner_type__200(self, user_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            print('Hello World!')
        login(client, 'admin')
        valid_filter_set_data_for_update[OWNER_TYPE_FIELD] = 'Dashboard'
        response = call_update_filter_set(client, user_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 200
        user_based_filter_set_dict['owner_id'] = user_based_filter_set_dict['dashboard_id']
        assert_filterset_updated(user_based_filter_set_dict, valid_filter_set_data_for_update)

    def test_when_caller_is_admin_and_owner_type_is_user__200(self, test_users: dict[str, int], user_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            return 10
        login(client, 'admin')
        response = call_update_filter_set(client, user_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 200
        assert_filterset_updated(user_based_filter_set_dict, valid_filter_set_data_for_update)

    def test_when_caller_is_admin_and_owner_type_is_dashboard__200(self, test_users: dict[str, int], dashboard_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            return 10
        login(client, 'admin')
        response = call_update_filter_set(client, dashboard_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 200
        assert_filterset_updated(dashboard_based_filter_set_dict, valid_filter_set_data_for_update)

    def test_when_caller_is_dashboard_owner_and_owner_is_other_user_403(self, test_users: dict[str, int], user_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            while True:
                i = 10
        login(client, DASHBOARD_OWNER_USERNAME)
        response = call_update_filter_set(client, user_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 403
        assert_filterset_was_not_updated(user_based_filter_set_dict)

    def test_when_caller_is_dashboard_owner_and_owner_type_is_dashboard__200(self, test_users: dict[str, int], dashboard_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            while True:
                i = 10
        login(client, DASHBOARD_OWNER_USERNAME)
        response = call_update_filter_set(client, dashboard_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 200
        assert_filterset_updated(dashboard_based_filter_set_dict, valid_filter_set_data_for_update)

    def test_when_caller_is_filterset_owner__200(self, test_users: dict[str, int], user_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            return 10
        login(client, FILTER_SET_OWNER_USERNAME)
        response = call_update_filter_set(client, user_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 200
        assert_filterset_updated(user_based_filter_set_dict, valid_filter_set_data_for_update)

    def test_when_caller_is_regular_user_and_owner_type_is_user__403(self, test_users: dict[str, int], user_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            for i in range(10):
                print('nop')
        login(client, REGULAR_USER)
        response = call_update_filter_set(client, user_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 403
        assert_filterset_was_not_updated(user_based_filter_set_dict)

    def test_when_caller_is_regular_user_and_owner_type_is_dashboard__403(self, test_users: dict[str, int], dashboard_based_filter_set_dict: dict[str, Any], valid_filter_set_data_for_update: dict[str, Any], client: FlaskClient[Any]):
        if False:
            return 10
        login(client, REGULAR_USER)
        response = call_update_filter_set(client, dashboard_based_filter_set_dict, valid_filter_set_data_for_update)
        assert response.status_code == 403
        assert_filterset_was_not_updated(dashboard_based_filter_set_dict)