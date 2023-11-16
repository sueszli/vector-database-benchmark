"""
Tests for salt-run with show_jid
"""
import logging
import re
import pytest
log = logging.getLogger(__name__)

@pytest.fixture(scope='module')
def salt_master(salt_factories):
    if False:
        i = 10
        return i + 15
    '\n    Salt master with `show_jid: True`\n    '
    config_defaults = {'show_jid': True}
    salt_master = salt_factories.salt_master_daemon('salt-run-show-jid-master', defaults=config_defaults)
    with salt_master.started():
        yield salt_master

@pytest.fixture(scope='module')
def salt_run_cli(salt_master):
    if False:
        i = 10
        return i + 15
    '\n    The ``salt-run`` CLI as a fixture against the running master\n    '
    assert salt_master.is_running()
    return salt_master.salt_run_cli(timeout=30)

def test_salt_run_show_jid(salt_run_cli):
    if False:
        i = 10
        return i + 15
    '\n    Test that jid is output\n    '
    ret = salt_run_cli.run('test.stdout_print')
    assert re.match('jid: \\d+', ret.stdout)