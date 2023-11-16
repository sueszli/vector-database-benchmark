from pathlib import Path
from typing import Optional
from tribler.core.config.tribler_config_section import TriblerConfigSection

class TriblerTestConfigSection(TriblerConfigSection):
    path: Optional[str]

def test_put_path_relative(tmpdir):
    if False:
        while True:
            i = 10
    section = TriblerTestConfigSection()
    section.put_path_as_relative(property_name='path', value=Path(tmpdir), state_dir=tmpdir)
    assert section.path == '.'
    section.put_path_as_relative(property_name='path', value=Path(tmpdir) / '1', state_dir=tmpdir)
    assert section.path == '1'

def test_put_path_absolute(tmpdir):
    if False:
        i = 10
        return i + 15
    section = TriblerTestConfigSection()
    section.put_path_as_relative(property_name='path')
    assert not section.path
    section.put_path_as_relative(property_name='path', value=Path(tmpdir).parent, state_dir=tmpdir)
    assert section.path == str(Path(tmpdir).parent)
    section.put_path_as_relative(property_name='path', value=Path('/Tribler'), state_dir=tmpdir)
    assert section.path == str(Path('/Tribler'))

def test_null_replacement():
    if False:
        return 10
    section = TriblerTestConfigSection(path='None')
    assert section.path is None