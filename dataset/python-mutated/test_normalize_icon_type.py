import os
import sys
import shutil
from pathlib import Path
import pytest
import PyInstaller
from PyInstaller.building.icon import normalize_icon_type

def test_normalize_icon(monkeypatch, tmp_path):
    if False:
        return 10
    workpath = str(tmp_path)
    icon = 'this_is_not_a_file.ico'
    with pytest.raises(FileNotFoundError):
        normalize_icon_type(icon, ('ico',), 'ico', workpath)
    icon = str(Path(PyInstaller.__file__).with_name('bootloader') / 'images' / 'icon-console.ico')
    ret = normalize_icon_type(icon, ('ico',), 'ico', workpath)
    if ret != icon:
        pytest.fail('icon validation changed path even though the format was correct already', False)
    monkeypatch.setitem(sys.modules, 'PIL', None)
    icon = str(Path(__file__, '../../functional/data/splash/image.png').resolve())
    assert os.path.exists(icon)
    with pytest.raises(ValueError):
        normalize_icon_type(icon, ('ico',), 'ico', workpath)

def test_normalize_icon_pillow(tmp_path):
    if False:
        return 10
    workpath = str(tmp_path)
    pytest.importorskip('PIL', reason='Needs PIL / Pillow for this test')
    icon = str(Path(__file__, '../../functional/data/splash/image.png').resolve())
    ret = normalize_icon_type(icon, ('ico',), 'ico', workpath)
    (_, ret_filetype) = os.path.splitext(ret)
    if ret_filetype != '.ico':
        pytest.fail("icon validation didn't convert to the right format", False)
    for (i, suffix) in enumerate(['ico', 'ICO']):
        png = shutil.copy(icon, str(tmp_path / f'png-in-disguise-{i}.{suffix}'))
        normalised = normalize_icon_type(png, ('exe', 'ico'), 'ico', workpath)
        assert normalised != png
        assert normalize_icon_type(normalised, ('exe', 'ico'), 'ico', workpath) == normalised
    icon = os.path.join(tmp_path, 'pyi_icon.notanicon')
    with open(icon, 'w') as f:
        f.write('this is in fact, not an icon')
    with pytest.raises(ValueError):
        normalize_icon_type(icon, ('ico',), 'ico', workpath)