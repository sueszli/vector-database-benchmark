"""
Tests for minion blackout
"""
import logging
import pytest
log = logging.getLogger(__name__)

def _check_skip(grains):
    if False:
        i = 10
        return i + 15
    if grains['os'] == 'Windows' and grains['osrelease'] == '2016Server':
        return True
    return False
pytestmark = [pytest.mark.slow_test, pytest.mark.windows_whitelisted, pytest.mark.skip_initial_gh_actions_failure(skip=_check_skip)]

def test_blackout(salt_cli, blackout, salt_minion_1):
    if False:
        i = 10
        return i + 15
    '\n    Test that basic minion blackout functionality works\n    '
    ret = salt_cli.run('test.ping', minion_tgt=salt_minion_1.id)
    assert ret.returncode == 0
    assert ret.data is True
    with blackout.enter_blackout('minion_blackout: true'):
        ret = salt_cli.run('test.ping', minion_tgt=salt_minion_1.id)
        assert ret.returncode == 1
        assert 'Minion in blackout mode.' in ret.stdout
    ret = salt_cli.run('test.ping', minion_tgt=salt_minion_1.id)
    assert ret.returncode == 0
    assert ret.data is True

def test_blackout_whitelist(salt_cli, blackout, salt_minion_1):
    if False:
        i = 10
        return i + 15
    '\n    Test that minion blackout whitelist works\n    '
    blackout_contents = '\n    minion_blackout: True\n    minion_blackout_whitelist:\n      - test.ping\n      - test.fib\n    '
    ret = salt_cli.run('test.ping', minion_tgt=salt_minion_1.id)
    assert ret.returncode == 0
    assert ret.data is True
    with blackout.enter_blackout(blackout_contents):
        ret = salt_cli.run('test.ping', minion_tgt=salt_minion_1.id)
        assert ret.returncode == 0
        assert ret.data is True
        ret = salt_cli.run('test.fib', '7', minion_tgt=salt_minion_1.id)
        assert ret.returncode == 0
        assert ret.data[0] == 13

def test_blackout_nonwhitelist(salt_cli, blackout, salt_minion_1):
    if False:
        i = 10
        return i + 15
    '\n    Test that minion refuses to run non-whitelisted functions during\n    blackout whitelist\n    '
    blackout_contents = '\n    minion_blackout: True\n    minion_blackout_whitelist:\n      - test.ping\n      - test.fib\n    '
    ret = salt_cli.run('test.ping', minion_tgt=salt_minion_1.id)
    assert ret.returncode == 0
    assert ret.data is True
    with blackout.enter_blackout(blackout_contents):
        ret = salt_cli.run('test.ping', minion_tgt=salt_minion_1.id)
        assert ret.returncode == 0
        assert ret.data is True
        ret = salt_cli.run('state.apply', minion_tgt=salt_minion_1.id)
        assert ret.returncode == 1
        assert 'Minion in blackout mode.' in ret.stdout
        ret = salt_cli.run('cloud.query', 'list_nodes_full', minion_tgt=salt_minion_1.id)
        assert ret.returncode == 1
        assert 'Minion in blackout mode.' in ret.stdout
    ret = salt_cli.run('test.ping', minion_tgt=salt_minion_1.id)
    assert ret.returncode == 0
    assert ret.data is True