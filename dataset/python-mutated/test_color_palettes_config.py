import importlib
import os
COLOR_PALETTES_CONFIG = "- - - '#5ff550'\n    - '#fae62d'\n    - '#f037a5'\n  - categorical\n  - Foo Bar\n- - - '#ffc864'\n    - '#ffcdd2'\n    - '#eb1e32'\n  - categorical\n  - Baz Quux\n"
EXPECTED_COLOR_PALETTES = {'foo bar': ['#5ff550', '#fae62d', '#f037a5'], 'baz quux': ['#ffc864', '#ffcdd2', '#eb1e32']}

def test_color_palettes_config(monkeypatch, tmpdir):
    if False:
        print('Hello World!')
    f = tmpdir.join('color_palettes_config.yaml')
    f.write(COLOR_PALETTES_CONFIG)
    monkeypatch.setenv('CHARTIFY_CONFIG_DIR', os.path.join(str(tmpdir), ''))
    import chartify._core.options
    import chartify._core.colors
    importlib.reload(chartify._core.options)
    importlib.reload(chartify._core.colors)
    color_palettes = chartify._core.colors.color_palettes
    for (name, palette) in EXPECTED_COLOR_PALETTES.items():
        assert color_palettes[name].to_hex_list() == palette