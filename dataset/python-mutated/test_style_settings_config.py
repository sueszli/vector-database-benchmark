import importlib
import os
STYLE_SETTINGS_CONFIG = 'foo:\n  baz.bar: 0.25\n  quux: deadbeef\nbar:\n  baz: bar quux\n'

def test_style_settings_config(monkeypatch, tmpdir):
    if False:
        while True:
            i = 10
    f = tmpdir.join('style_settings_config.yaml')
    f.write(STYLE_SETTINGS_CONFIG)
    monkeypatch.setenv('CHARTIFY_CONFIG_DIR', os.path.join(str(tmpdir), ''))
    import chartify._core.options
    import chartify._core.style
    importlib.reload(chartify._core.options)
    importlib.reload(chartify._core.style)
    style = chartify._core.style.Style(None, '')
    import yaml
    expected_settings = yaml.safe_load(STYLE_SETTINGS_CONFIG)
    for (key, expected_value) in expected_settings.items():
        actual_value = style.settings[key]
        assert expected_value == actual_value