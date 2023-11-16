import importlib
import os
COLORS_CONFIG = '? !!python/tuple\n- 0\n- 100\n- 80\n: Foo\n? !!python/tuple\n- 25\n- 20\n- 20\n: Bar\n? !!python/tuple\n- 25\n- 230\n- 140\n: Baz\n'
EXPECTED_COLORS = {'style.color_palette_categorical': 'My Palette', 'style.color_palette_sequential': 'Midnight Orange Sequential', 'style.color_palette_diverging': 'Midnight Orange Diverging', 'style.color_palette_accent': 'My Palette', 'style.color_palette_accent_default_color': 'light grey'}

def test_colors_config(monkeypatch, tmpdir):
    if False:
        print('Hello World!')
    f = tmpdir.join('colors_config.yaml')
    f.write(COLORS_CONFIG)
    monkeypatch.setenv('CHARTIFY_CONFIG_DIR', os.path.join(str(tmpdir), ''))
    import chartify._core.options
    import chartify._core.colors
    import chartify._core.style
    importlib.reload(chartify._core.options)
    importlib.reload(chartify._core.colors)
    import chartify._core.colour as colour
    assert colour.COLOR_NAME_TO_RGB['foo'] == (0, 100, 80)
    assert colour.COLOR_NAME_TO_RGB['bar'] == (25, 20, 20)
    assert colour.COLOR_NAME_TO_RGB['baz'] == (25, 230, 140)