import pytest
import salt.client.ssh.client
import salt.config
import salt.roster
import salt.utils.files
import salt.utils.path
import salt.utils.thin
import salt.utils.yaml
from tests.support.mock import MagicMock, patch

@pytest.fixture
def roster():
    if False:
        for i in range(10):
            print('nop')
    return '\n    localhost:\n        host: 127.0.0.1\n        port: 2827\n    self:\n        host: 0.0.0.0\n        port: 42\n    '

def test_roster_defaults_flat(tmp_path, roster):
    if False:
        while True:
            i = 10
    '\n    Test Roster Defaults on the flat roster\n    '
    expected = {'self': {'host': '0.0.0.0', 'user': 'daniel', 'port': 42}, 'localhost': {'host': '127.0.0.1', 'user': 'daniel', 'port': 2827}}
    root_dir = tmp_path / 'foo' / 'bar'
    root_dir.mkdir(exist_ok=True, parents=True)
    fpath = root_dir / 'config'
    with salt.utils.files.fopen(str(fpath), 'w') as fp_:
        fp_.write('\n            roster_defaults:\n                user: daniel\n            ')
    opts = salt.config.master_config(fpath)
    with patch('salt.roster.get_roster_file', MagicMock(return_value=roster)):
        with patch('salt.template.compile_template', MagicMock(return_value=salt.utils.yaml.safe_load(roster))):
            roster = salt.roster.Roster(opts=opts)
            assert roster.targets('*', 'glob') == expected