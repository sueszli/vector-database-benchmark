"""
:codeauthor: Gareth J. Greenaway (ggreenaway@vmware.com)
"""
import logging
import os
import random
import pytest
from saltfactories.utils import random_string
import salt.defaults.exitcodes
from tests.support.helpers import PRE_PYTEST_SKIP_REASON
log = logging.getLogger(__name__)
pytestmark = [pytest.mark.skip_on_spawning_platform(reason='Deltaproxy minions do not currently work on spawning platforms.'), pytest.mark.core_test]

@pytest.fixture(scope='package')
def salt_master(salt_factories):
    if False:
        while True:
            i = 10
    config_defaults = {'open_mode': True}
    salt_master = salt_factories.salt_master_daemon('deltaproxy-functional-master', defaults=config_defaults)
    with salt_master.started():
        yield salt_master

@pytest.fixture(scope='package')
def salt_cli(salt_master):
    if False:
        print('Hello World!')
    '\n    The ``salt`` CLI as a fixture against the running master\n    '
    assert salt_master.is_running()
    return salt_master.salt_cli(timeout=30)

@pytest.fixture(scope='package', autouse=True)
def skip_on_tcp_transport(request):
    if False:
        return 10
    if request.config.getoption('--transport') == 'tcp':
        pytest.skip('Deltaproxy under the TPC transport is not working. See #61367')

@pytest.fixture
def proxy_minion_id(salt_master):
    if False:
        i = 10
        return i + 15
    _proxy_minion_id = random_string('proxy-minion-')
    try:
        yield _proxy_minion_id
    finally:
        pytest.helpers.remove_stale_minion_key(salt_master, _proxy_minion_id)

def clear_proxy_minions(salt_master, proxy_minion_id):
    if False:
        i = 10
        return i + 15
    for proxy in [proxy_minion_id, 'dummy_proxy_one', 'dummy_proxy_two']:
        pytest.helpers.remove_stale_minion_key(salt_master, proxy)
        cachefile = os.path.join(salt_master.config['cachedir'], '{}.cache'.format(proxy))
        if os.path.exists(cachefile):
            os.unlink(cachefile)

@pytest.mark.skip_on_windows(reason=PRE_PYTEST_SKIP_REASON)
@pytest.mark.parametrize('parallel_startup', [True, False], ids=['parallel_startup=True', 'parallel_startup=False'])
def test_exit_status_correct_usage_large_number_of_minions(salt_master, salt_cli, proxy_minion_id, parallel_startup):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure the salt-proxy control proxy starts and\n    is able to respond to test.ping, additionally ensure that\n    the proxies being controlled also respond to test.ping.\n\n    Finally ensure correct exit status when salt-proxy exits correctly.\n\n    Skip on Windows because daemonization not supported\n    '
    config_defaults = {'metaproxy': 'deltaproxy'}
    sub_proxies = ['proxy_one', 'proxy_two', 'proxy_three', 'proxy_four', 'proxy_five', 'proxy_six', 'proxy_seven', 'proxy_eight', 'proxy_nine', 'proxy_ten', 'proxy_eleven', 'proxy_twelve', 'proxy_thirteen', 'proxy_fourteen', 'proxy_fifteen', 'proxy_sixteen', 'proxy_seventeen', 'proxy_eighteen', 'proxy_nineteen', 'proxy_twenty', 'proxy_twenty_one', 'proxy_twenty_two', 'proxy_twenty_three', 'proxy_twenty_four', 'proxy_twenty_five', 'proxy_twenty_six', 'proxy_twenty_seven', 'proxy_twenty_eight', 'proxy_twenty_nine', 'proxy_thirty', 'proxy_thirty_one', 'proxy_thirty_two']
    top_file = '\n    base:\n      {control}:\n        - controlproxy\n    '.format(control=proxy_minion_id)
    controlproxy_pillar_file = '\n    proxy:\n        proxytype: deltaproxy\n        parallel_startup: {}\n        ids:\n    '.format(parallel_startup)
    dummy_proxy_pillar_file = '\n    proxy:\n      proxytype: dummy\n    '
    for minion_id in sub_proxies:
        top_file += '\n      {minion_id}:\n        - dummy'.format(minion_id=minion_id)
        controlproxy_pillar_file += '\n            - {}\n        '.format(minion_id)
    top_tempfile = salt_master.pillar_tree.base.temp_file('top.sls', top_file)
    controlproxy_tempfile = salt_master.pillar_tree.base.temp_file('controlproxy.sls', controlproxy_pillar_file)
    dummy_proxy_tempfile = salt_master.pillar_tree.base.temp_file('dummy.sls', dummy_proxy_pillar_file)
    with top_tempfile, controlproxy_tempfile, dummy_proxy_tempfile:
        with salt_master.started():
            assert salt_master.is_running()
            factory = salt_master.salt_proxy_minion_daemon(proxy_minion_id, defaults=config_defaults, extra_cli_arguments_after_first_start_failure=['--log-level=info'], start_timeout=240)
            for minion_id in [proxy_minion_id] + sub_proxies:
                factory.before_start(pytest.helpers.remove_stale_proxy_minion_cache_file, factory, minion_id)
                factory.after_terminate(pytest.helpers.remove_stale_minion_key, salt_master, minion_id)
                factory.after_terminate(pytest.helpers.remove_stale_proxy_minion_cache_file, factory, minion_id)
            with factory.started():
                assert factory.is_running()
                ret = salt_cli.run('test.ping', minion_tgt=proxy_minion_id)
                assert ret.returncode == 0
                assert ret.data is True
                for minion_id in random.sample(sub_proxies, 4):
                    ret = salt_cli.run('test.ping', minion_tgt=minion_id)
                    assert ret.returncode == 0
                    assert ret.data is True
        ret = factory.terminate()
        assert ret.returncode == salt.defaults.exitcodes.EX_OK, ret
        ret = salt_master.terminate()
        assert ret.returncode == salt.defaults.exitcodes.EX_OK, ret