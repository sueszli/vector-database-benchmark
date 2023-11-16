import json
from superset.app import SupersetApp
from superset.migrations.shared.migrate_viz import MigrateTreeMap
treemap_form_data = '{\n  "adhoc_filters": [\n    {\n      "clause": "WHERE",\n      "comparator": [\n        "Edward"\n      ],\n      "expressionType": "SIMPLE",\n      "filterOptionName": "filter_xhbus6irfa_r10k9nwmwy",\n      "isExtra": false,\n      "isNew": false,\n      "operator": "IN",\n      "operatorId": "IN",\n      "sqlExpression": null,\n      "subject": "name"\n    }\n  ],\n  "color_scheme": "bnbColors",\n  "datasource": "2__table",\n  "extra_form_data": {},\n  "granularity_sqla": "ds",\n  "groupby": [\n    "state",\n    "gender"\n  ],\n  "metrics": [\n    "sum__num"\n  ],\n  "number_format": ",d",\n  "order_desc": true,\n  "row_limit": 10,\n  "time_range": "No filter",\n  "timeseries_limit_metric": "sum__num",\n  "treemap_ratio": 1.618033988749895,\n  "viz_type": "treemap"\n}\n'

def test_treemap_migrate(app_context: SupersetApp) -> None:
    if False:
        for i in range(10):
            print('nop')
    from superset.models.slice import Slice
    slc = Slice(viz_type=MigrateTreeMap.source_viz_type, datasource_type='table', params=treemap_form_data, query_context=f'{{"form_data": {treemap_form_data}}}')
    slc = MigrateTreeMap.upgrade_slice(slc)
    assert slc.viz_type == MigrateTreeMap.target_viz_type
    new_form_data = json.loads(slc.params)
    assert new_form_data['metric'] == 'sum__num'
    assert new_form_data['viz_type'] == 'treemap_v2'
    assert 'metrics' not in new_form_data
    assert json.dumps(new_form_data['form_data_bak'], sort_keys=True) == json.dumps(json.loads(treemap_form_data), sort_keys=True)
    new_query_context = json.loads(slc.query_context)
    assert new_query_context['form_data']['viz_type'] == 'treemap_v2'
    slc = MigrateTreeMap.downgrade_slice(slc)
    assert slc.viz_type == MigrateTreeMap.source_viz_type
    assert json.dumps(json.loads(slc.params), sort_keys=True) == json.dumps(json.loads(treemap_form_data), sort_keys=True)