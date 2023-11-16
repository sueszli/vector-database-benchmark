import json
from superset.app import SupersetApp
from superset.migrations.shared.migrate_viz import MigrateAreaChart
area_form_data = '{\n  "adhoc_filters": [],\n  "annotation_layers": [],\n  "bottom_margin": "auto",\n  "color_scheme": "lyftColors",\n  "comparison_type": "values",\n  "contribution": true,\n  "datasource": "2__table",\n  "extra_form_data": {},\n  "granularity_sqla": "ds",\n  "groupby": [\n    "gender"\n  ],\n  "line_interpolation": "linear",\n  "metrics": [\n    "sum__num"\n  ],\n  "order_desc": true,\n  "rich_tooltip": true,\n  "rolling_type": "None",\n  "row_limit": 10000,\n  "show_brush": "auto",\n  "show_controls": true,\n  "show_legend": true,\n  "slice_id": 165,\n  "stacked_style": "stack",\n  "time_grain_sqla": "P1D",\n  "time_range": "No filter",\n  "viz_type": "area",\n  "x_axis_format": "smart_date",\n  "x_axis_label": "x asix label",\n  "x_axis_showminmax": false,\n  "x_ticks_layout": "auto",\n  "y_axis_bounds": [\n    null,\n    null\n  ],\n  "y_axis_format": "SMART_NUMBER"\n}\n'

def test_area_migrate(app_context: SupersetApp) -> None:
    if False:
        return 10
    from superset.models.slice import Slice
    slc = Slice(viz_type=MigrateAreaChart.source_viz_type, datasource_type='table', params=area_form_data, query_context=f'{{"form_data": {area_form_data}}}')
    slc = MigrateAreaChart.upgrade_slice(slc)
    assert slc.viz_type == MigrateAreaChart.target_viz_type
    new_form_data = json.loads(slc.params)
    assert new_form_data['contributionMode'] == 'row'
    assert 'contribution' not in new_form_data
    assert new_form_data['show_extra_controls'] is True
    assert new_form_data['stack'] == 'Stack'
    assert new_form_data['x_axis_title'] == 'x asix label'
    assert new_form_data['x_axis_title_margin'] == 30
    assert json.dumps(new_form_data['form_data_bak'], sort_keys=True) == json.dumps(json.loads(area_form_data), sort_keys=True)
    new_query_context = json.loads(slc.query_context)
    assert new_query_context['form_data']['viz_type'] == MigrateAreaChart.target_viz_type
    slc = MigrateAreaChart.downgrade_slice(slc)
    assert slc.viz_type == MigrateAreaChart.source_viz_type
    assert json.dumps(json.loads(slc.params), sort_keys=True) == json.dumps(json.loads(area_form_data), sort_keys=True)