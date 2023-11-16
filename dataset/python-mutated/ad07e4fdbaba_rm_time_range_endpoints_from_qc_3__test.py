import json
from importlib import import_module
rm_time_range_endpoints_from_qc_3 = import_module('superset.migrations.versions.2022-04-18_11-20_ad07e4fdbaba_rm_time_range_endpoints_from_qc_3')
Slice = rm_time_range_endpoints_from_qc_3.Slice
upgrade_slice = rm_time_range_endpoints_from_qc_3.upgrade_slice
sample_query_context = {'datasource': {'id': 27, 'type': 'table'}, 'force': False, 'queries': [{'time_range': 'No filter', 'filters': [], 'extras': {'time_grain_sqla': 'P1D', 'time_range_endpoints': ['inclusive', 'exclusive'], 'having': '', 'where': ''}, 'applied_time_extras': {}, 'columns': ['a', 'b'], 'orderby': [], 'annotation_layers': [], 'row_limit': 1000, 'timeseries_limit': 0, 'order_desc': True, 'url_params': {}, 'custom_params': {}, 'custom_form_data': {}, 'post_processing': []}], 'form_data': {'viz_type': 'table', 'datasource': '27__table', 'slice_id': 545, 'url_params': {}, 'time_grain_sqla': 'P1D', 'time_range': 'No filter', 'query_mode': 'raw', 'groupby': [], 'metrics': [], 'all_columns': ['a', 'b'], 'percent_metrics': [], 'adhoc_filters': [], 'order_by_cols': [], 'row_limit': 1000, 'server_page_length': 10, 'include_time': False, 'order_desc': True, 'table_timestamp_format': 'smart_date', 'show_cell_bars': True, 'color_pn': True, 'extra_form_data': {}, 'force': False, 'result_format': 'json', 'result_type': 'full'}, 'result_format': 'json', 'result_type': 'full'}
sample_query_context = {'datasource': {'id': 27, 'type': 'table'}, 'force': False, 'queries': [{'time_range': 'No filter', 'filters': [], 'extras': {'time_grain_sqla': 'P1D', 'time_range_endpoints': ['inclusive', 'exclusive'], 'having': '', 'where': ''}, 'applied_time_extras': {}, 'columns': ['a', 'b'], 'orderby': [], 'annotation_layers': [], 'row_limit': 1000, 'timeseries_limit': 0, 'order_desc': True, 'url_params': {}, 'custom_params': {}, 'custom_form_data': {}, 'post_processing': []}], 'form_data': {'time_range_endpoints': ['inclusive', 'exclusive']}, 'result_format': 'json', 'result_type': 'full'}

def test_upgrade():
    if False:
        print('Hello World!')
    slc = Slice(slice_name='FOO', query_context=json.dumps(sample_query_context))
    upgrade_slice(slc)
    query_context = json.loads(slc.query_context)
    queries = query_context.get('queries')
    for q in queries:
        extras = q.get('extras', {})
        assert 'time_range_endpoints' not in extras
    form_data = query_context.get('form_data', {})
    assert 'time_range_endpoints' not in form_data

def test_upgrade_bad_json():
    if False:
        return 10
    slc = Slice(slice_name='FOO', query_context='abc')
    assert None == upgrade_slice(slc)