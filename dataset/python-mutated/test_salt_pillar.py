import pytest
pytestmark = [pytest.mark.skip_on_windows]

def test_salt_pillar(salt_cli, salt_minion):
    if False:
        i = 10
        return i + 15
    '\n    Test pillar.items\n    '
    ret = salt_cli.run('pillar.items', minion_tgt=salt_minion.id)
    assert 'info' in ret.data