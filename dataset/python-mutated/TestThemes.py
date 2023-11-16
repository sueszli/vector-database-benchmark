import json
import os
import pytest
theme_base = os.path.join(os.path.split(__file__)[0], '..', 'resources', 'themes')
theme_paths = [os.path.join(theme_base, theme_folder) for theme_folder in os.listdir(theme_base) if os.path.isdir(os.path.join(theme_base, theme_folder))]

@pytest.mark.parametrize('theme_path', theme_paths)
def test_deprecatedIconsExist(theme_path: str) -> None:
    if False:
        while True:
            i = 10
    icons_folder = os.path.join(theme_path, 'icons')
    deprecated_icons_file = os.path.join(icons_folder, 'deprecated_icons.json')
    if not os.path.exists(deprecated_icons_file):
        return
    existing_icons = {}
    for size in [subfolder for subfolder in os.listdir(icons_folder) if os.path.isdir(os.path.join(icons_folder, subfolder))]:
        existing_icons[size] = set((os.path.splitext(fname)[0] for fname in os.listdir(os.path.join(icons_folder, size))))
    with open(deprecated_icons_file) as f:
        deprecated_icons = json.load(f)
    for entry in deprecated_icons.values():
        assert 'new_icon' in entry
        new_icon = entry['new_icon']
        assert 'size' in entry
        size = entry['size']
        assert size in existing_icons
        assert new_icon in existing_icons[size]