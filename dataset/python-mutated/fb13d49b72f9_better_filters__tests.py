import json
from importlib import import_module
better_filters = import_module('superset.migrations.versions.2018-12-11_22-03_fb13d49b72f9_better_filters')
Slice = better_filters.Slice
upgrade_slice = better_filters.upgrade_slice

def test_upgrade_slice():
    if False:
        for i in range(10):
            print('nop')
    slc = Slice(slice_name='FOO', viz_type='filter_box', params=json.dumps(dict(metric='foo', groupby=['bar'])))
    upgrade_slice(slc)
    params = json.loads(slc.params)
    assert 'metric' not in params
    assert 'filter_configs' in params
    cfg = params['filter_configs'][0]
    assert cfg.get('metric') == 'foo'