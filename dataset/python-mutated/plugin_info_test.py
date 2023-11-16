import sys
from io import StringIO
from pathlib import Path
import pytest
from errbot.plugin_info import PluginInfo
plugfile_base = Path(__file__).absolute().parent / 'config_plugin'
plugfile_path = plugfile_base / 'config.plug'

def test_load_from_plugfile_path():
    if False:
        i = 10
        return i + 15
    pi = PluginInfo.load(plugfile_path)
    assert pi.name == 'Config'
    assert pi.module == 'config'
    assert pi.doc is None
    assert pi.python_version == (3, 0, 0)
    assert pi.errbot_minversion is None
    assert pi.errbot_maxversion is None

@pytest.mark.parametrize('test_input,expected', [('2', (2, 0, 0)), ('2+', (3, 0, 0)), ('3', (3, 0, 0)), ('1.2.3', (1, 2, 3)), ('1.2.3-beta', (1, 2, 3))])
def test_python_version_parse(test_input, expected):
    if False:
        return 10
    f = StringIO('\n    [Core]\n    Name = Config\n    Module = config\n\n    [Python]\n    Version = %s\n    ' % test_input)
    assert PluginInfo.load_file(f, None).python_version == expected

def test_doc():
    if False:
        while True:
            i = 10
    f = StringIO('\n    [Core]\n    Name = Config\n    Module = config\n\n    [Documentation]\n    Description = something\n    ')
    assert PluginInfo.load_file(f, None).doc == 'something'

def test_errbot_version():
    if False:
        return 10
    f = StringIO('\n    [Core]\n    Name = Config\n    Module = config\n    [Errbot]\n    Min = 1.2.3\n    Max = 4.5.6-beta\n    ')
    info = PluginInfo.load_file(f, None)
    assert info.errbot_minversion == (1, 2, 3, sys.maxsize)
    assert info.errbot_maxversion == (4, 5, 6, 0)