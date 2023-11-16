import json
from superset.migrations.shared.migrate_viz import MigratePivotTable
from tests.unit_tests.conftest import with_feature_flags
SOURCE_FORM_DATA = {'adhoc_filters': [], 'any_other_key': 'untouched', 'columns': ['state'], 'combine_metric': True, 'granularity_sqla': 'ds', 'groupby': ['name'], 'number_format': 'SMART_NUMBER', 'pandas_aggfunc': 'sum', 'pivot_margins': True, 'time_range': '100 years ago : now', 'timeseries_limit_metric': 'count', 'transpose_pivot': True, 'viz_type': 'pivot_table'}
TARGET_FORM_DATA = {'adhoc_filters': [], 'any_other_key': 'untouched', 'aggregateFunction': 'Sum', 'colTotals': True, 'colSubTotals': True, 'combineMetric': True, 'form_data_bak': SOURCE_FORM_DATA, 'granularity_sqla': 'ds', 'groupbyColumns': ['state'], 'groupbyRows': ['name'], 'rowOrder': 'value_z_to_a', 'series_limit_metric': 'count', 'time_range': '100 years ago : now', 'transposePivot': True, 'valueFormat': 'SMART_NUMBER', 'viz_type': 'pivot_table_v2'}

@with_feature_flags(GENERIC_CHART_AXES=False)
def test_migration_without_generic_chart_axes() -> None:
    if False:
        i = 10
        return i + 15
    source = SOURCE_FORM_DATA.copy()
    target = TARGET_FORM_DATA.copy()
    upgrade_downgrade(source, target)

@with_feature_flags(GENERIC_CHART_AXES=True)
def test_migration_with_generic_chart_axes() -> None:
    if False:
        while True:
            i = 10
    source = SOURCE_FORM_DATA.copy()
    target = TARGET_FORM_DATA.copy()
    target['adhoc_filters'] = [{'clause': 'WHERE', 'comparator': '100 years ago : now', 'expressionType': 'SIMPLE', 'operator': 'TEMPORAL_RANGE', 'subject': 'ds'}]
    target.pop('granularity_sqla')
    target.pop('time_range')
    upgrade_downgrade(source, target)

@with_feature_flags(GENERIC_CHART_AXES=True)
def test_custom_sql_time_column() -> None:
    if False:
        print('Hello World!')
    source = SOURCE_FORM_DATA.copy()
    source['granularity_sqla'] = {'expressionType': 'SQL', 'label': 'ds', 'sqlExpression': 'sum(ds)'}
    target = TARGET_FORM_DATA.copy()
    target['adhoc_filters'] = [{'clause': 'WHERE', 'comparator': None, 'expressionType': 'SQL', 'operator': 'TEMPORAL_RANGE', 'sqlExpression': 'sum(ds)', 'subject': 'ds'}]
    target['form_data_bak'] = source
    target.pop('granularity_sqla')
    target.pop('time_range')
    upgrade_downgrade(source, target)

def upgrade_downgrade(source, target) -> None:
    if False:
        i = 10
        return i + 15
    from superset.models.slice import Slice
    dumped_form_data = json.dumps(source)
    slc = Slice(viz_type=MigratePivotTable.source_viz_type, datasource_type='table', params=dumped_form_data, query_context=f'{{"form_data": {dumped_form_data}}}')
    slc = MigratePivotTable.upgrade_slice(slc)
    new_form_data = json.loads(slc.params)
    assert new_form_data == target
    assert new_form_data['form_data_bak'] == source
    new_query_context = json.loads(slc.query_context)
    assert new_query_context['form_data']['viz_type'] == 'pivot_table_v2'
    slc = MigratePivotTable.downgrade_slice(slc)
    assert slc.viz_type == MigratePivotTable.source_viz_type
    assert json.loads(slc.params) == source