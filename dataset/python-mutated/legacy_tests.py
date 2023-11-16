import copy
from typing import Any
from superset.legacy import update_time_range
from tests.unit_tests.conftest import with_feature_flags
original_form_data = {'granularity_sqla': 'order_date', 'datasource': '22__table', 'viz_type': 'table', 'query_mode': 'raw', 'groupby': [], 'time_grain_sqla': 'P1D', 'temporal_columns_lookup': {'order_date': True}, 'all_columns': ['order_date', 'state', 'product_code'], 'percent_metrics': [], 'adhoc_filters': [{'clause': 'WHERE', 'subject': 'order_date', 'operator': 'TEMPORAL_RANGE', 'comparator': 'No filter', 'expressionType': 'SIMPLE'}], 'order_by_cols': [], 'row_limit': 1000, 'server_page_length': 10, 'order_desc': True, 'table_timestamp_format': 'smart_date', 'show_cell_bars': True, 'color_pn': True, 'extra_form_data': {}, 'dashboards': [19], 'force': False, 'result_format': 'json', 'result_type': 'full', 'include_time': False}

def test_update_time_range_since_until() -> None:
    if False:
        i = 10
        return i + 15
    '\n    Tests for the old `since` and `until` parameters.\n    '
    form_data: dict[str, Any]
    form_data = {}
    update_time_range(form_data)
    assert form_data == {}
    form_data = {'since': 'yesterday'}
    update_time_range(form_data)
    assert form_data == {'time_range': 'yesterday : '}
    form_data = {'until': 'tomorrow'}
    update_time_range(form_data)
    assert form_data == {'time_range': ' : tomorrow'}
    form_data = {'since': 'yesterday', 'until': 'tomorrow'}
    update_time_range(form_data)
    assert form_data == {'time_range': 'yesterday : tomorrow'}

@with_feature_flags(GENERIC_CHART_AXES=False)
def test_update_time_range_granularity_sqla_no_feature_flag() -> None:
    if False:
        i = 10
        return i + 15
    '\n    Tests for the unfiltered `granularity_sqla` when `GENERIC_CHART_AXES` is off.\n    '
    form_data = copy.deepcopy(original_form_data)
    update_time_range(form_data)
    assert form_data == original_form_data

@with_feature_flags(GENERIC_CHART_AXES=True)
def test_update_time_range_granularity_sqla_with_feature_flag() -> None:
    if False:
        return 10
    '\n    Tests for the unfiltered `granularity_sqla` when `GENERIC_CHART_AXES` is on.\n    '
    form_data = copy.deepcopy(original_form_data)
    update_time_range(form_data)
    assert form_data['time_range'] == 'No filter'