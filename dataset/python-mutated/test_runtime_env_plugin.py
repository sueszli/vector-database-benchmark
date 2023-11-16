import asyncio
import logging
import os
from pathlib import Path
import time
from unittest import mock
import tempfile
import json
from typing import List
import pytest
import ray
from ray._private.runtime_env.context import RuntimeEnvContext
from ray._private.runtime_env.plugin import RuntimeEnvPlugin
from ray._private.test_utils import enable_external_redis, wait_for_condition
from ray.exceptions import RuntimeEnvSetupError
from ray.runtime_env.runtime_env import RuntimeEnv
MY_PLUGIN_CLASS_PATH = 'ray.tests.test_runtime_env_plugin.MyPlugin'
MY_PLUGIN_NAME = 'MyPlugin'

class MyPlugin(RuntimeEnvPlugin):
    name = MY_PLUGIN_NAME
    env_key = 'MY_PLUGIN_TEST_ENVIRONMENT_KEY'

    @staticmethod
    def validate(runtime_env: RuntimeEnv) -> str:
        if False:
            for i in range(10):
                print('nop')
        value = runtime_env[MY_PLUGIN_NAME]
        if value == 'fail':
            raise ValueError('not allowed')
        return value

    def modify_context(self, uris: List[str], runtime_env: RuntimeEnv, ctx: RuntimeEnvContext, logger: logging.Logger) -> None:
        if False:
            i = 10
            return i + 15
        plugin_config_dict = runtime_env[MY_PLUGIN_NAME]
        ctx.env_vars[MyPlugin.env_key] = str(plugin_config_dict['env_value'])
        ctx.command_prefix += ['echo', plugin_config_dict['tmp_content'], '>', plugin_config_dict['tmp_file'], '&&']
        ctx.py_executable = plugin_config_dict['prefix_command'] + ' ' + ctx.py_executable

@pytest.mark.parametrize('set_runtime_env_plugins', ['[{"class":"' + MY_PLUGIN_CLASS_PATH + '"}]'], indirect=True)
def test_simple_env_modification_plugin(set_runtime_env_plugins, ray_start_regular):
    if False:
        i = 10
        return i + 15
    (_, tmp_file_path) = tempfile.mkstemp()

    @ray.remote
    def f():
        if False:
            while True:
                i = 10
        import psutil
        with open(tmp_file_path, 'r') as f:
            content = f.read().strip()
        return {'env_value': os.environ[MyPlugin.env_key], 'tmp_content': content, 'nice': psutil.Process().nice()}
    with pytest.raises(RuntimeEnvSetupError, match='not allowed'):
        ray.get(f.options(runtime_env={MY_PLUGIN_NAME: 'fail'}).remote())
    if os.name != 'nt':
        output = ray.get(f.options(runtime_env={MY_PLUGIN_NAME: {'env_value': 42, 'tmp_file': tmp_file_path, 'tmp_content': 'hello', 'prefix_command': 'nice -n 19'}}).remote())
        assert output == {'env_value': '42', 'tmp_content': 'hello', 'nice': 19}
MY_PLUGIN_FOR_HANG_CLASS_PATH = 'ray.tests.test_runtime_env_plugin.MyPluginForHang'
MY_PLUGIN_FOR_HANG_NAME = 'MyPluginForHang'
my_plugin_setup_times = 0

class MyPluginForHang(RuntimeEnvPlugin):
    name = MY_PLUGIN_FOR_HANG_NAME
    env_key = 'MY_PLUGIN_FOR_HANG_TEST_ENVIRONMENT_KEY'

    @staticmethod
    def validate(runtime_env_dict: dict) -> str:
        if False:
            print('Hello World!')
        return 'True'

    async def create(self, uri: str, runtime_env: dict, ctx: RuntimeEnvContext, logger: logging.Logger) -> float:
        global my_plugin_setup_times
        my_plugin_setup_times += 1
        if my_plugin_setup_times == 1:
            await asyncio.sleep(3600)

    def modify_context(self, uris: List[str], plugin_config_dict: dict, ctx: RuntimeEnvContext, logger: logging.Logger) -> None:
        if False:
            print('Hello World!')
        global my_plugin_setup_times
        ctx.env_vars[MyPluginForHang.env_key] = str(my_plugin_setup_times)

@pytest.mark.parametrize('set_runtime_env_plugins', ['[{"class":"' + MY_PLUGIN_FOR_HANG_CLASS_PATH + '"}]'], indirect=True)
def test_plugin_hang(set_runtime_env_plugins, ray_start_regular):
    if False:
        for i in range(10):
            print('nop')
    env_key = MyPluginForHang.env_key

    @ray.remote(num_cpus=0.1)
    def f():
        if False:
            return 10
        return os.environ[env_key]
    refs = [f.options(runtime_env={MY_PLUGIN_FOR_HANG_NAME: {'name': 'f1'}}).remote(), f.options(runtime_env={MY_PLUGIN_FOR_HANG_NAME: {'name': 'f2'}}).remote()]

    def condition():
        if False:
            return 10
        for ref in refs:
            try:
                res = ray.get(ref, timeout=1)
                print('result:', res)
                assert int(res) == 2
                return True
            except Exception as error:
                print(f'Got error: {error}')
                pass
        return False
    wait_for_condition(condition, timeout=60)
DUMMY_PLUGIN_CLASS_PATH = 'ray.tests.test_runtime_env_plugin.DummyPlugin'
DUMMY_PLUGIN_NAME = 'DummyPlugin'
HANG_PLUGIN_CLASS_PATH = 'ray.tests.test_runtime_env_plugin.HangPlugin'
HANG_PLUGIN_NAME = 'HangPlugin'

class DummyPlugin(RuntimeEnvPlugin):
    name = DUMMY_PLUGIN_NAME

    @staticmethod
    def validate(runtime_env_dict: dict) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 1

class HangPlugin(DummyPlugin):
    name = HANG_PLUGIN_NAME

    async def create(self, uri: str, runtime_env: 'RuntimeEnv', ctx: RuntimeEnvContext, logger: logging.Logger) -> float:
        await asyncio.sleep(3600)

@pytest.mark.parametrize('set_runtime_env_plugins', ['[{"class":"' + DUMMY_PLUGIN_CLASS_PATH + '"},{"class":"' + HANG_PLUGIN_CLASS_PATH + '"}]'], indirect=True)
@pytest.mark.skipif(enable_external_redis(), reason='Failing in redis mode.')
def test_plugin_timeout(set_runtime_env_plugins, start_cluster):
    if False:
        for i in range(10):
            print('nop')

    @ray.remote(num_cpus=0.1)
    def f():
        if False:
            return 10
        return True
    refs = [f.options(runtime_env={HANG_PLUGIN_NAME: {'name': 'f1'}, 'config': {'setup_timeout_seconds': 1}}).remote(), f.options(runtime_env={DUMMY_PLUGIN_NAME: {'name': 'f2'}}).remote(), f.options(runtime_env={HANG_PLUGIN_NAME: {'name': 'f3'}, 'config': {'setup_timeout_seconds': -1}}).remote()]

    def condition():
        if False:
            while True:
                i = 10
        good_fun_num = 0
        bad_fun_num = 0
        for ref in refs:
            try:
                res = ray.get(ref, timeout=1)
                print('result:', res)
                if res:
                    good_fun_num += 1
                return True
            except RuntimeEnvSetupError:
                bad_fun_num += 1
        return bad_fun_num == 1 and good_fun_num == 2
    wait_for_condition(condition, timeout=60)
PRIORITY_TEST_PLUGIN1_CLASS_PATH = 'ray.tests.test_runtime_env_plugin.PriorityTestPlugin1'
PRIORITY_TEST_PLUGIN1_NAME = 'PriorityTestPlugin1'
PRIORITY_TEST_PLUGIN2_CLASS_PATH = 'ray.tests.test_runtime_env_plugin.PriorityTestPlugin2'
PRIORITY_TEST_PLUGIN2_NAME = 'PriorityTestPlugin2'
PRIORITY_TEST_ENV_VAR_NAME = 'PriorityTestEnv'

class PriorityTestPlugin1(RuntimeEnvPlugin):
    name = PRIORITY_TEST_PLUGIN1_NAME
    priority = 11
    env_value = ' world'

    @staticmethod
    def validate(runtime_env_dict: dict) -> str:
        if False:
            print('Hello World!')
        return None

    def modify_context(self, uris: List[str], plugin_config_dict: dict, ctx: RuntimeEnvContext, logger: logging.Logger) -> None:
        if False:
            while True:
                i = 10
        if PRIORITY_TEST_ENV_VAR_NAME in ctx.env_vars:
            ctx.env_vars[PRIORITY_TEST_ENV_VAR_NAME] += PriorityTestPlugin1.env_value
        else:
            ctx.env_vars[PRIORITY_TEST_ENV_VAR_NAME] = PriorityTestPlugin1.env_value

class PriorityTestPlugin2(RuntimeEnvPlugin):
    name = PRIORITY_TEST_PLUGIN2_NAME
    priority = 10
    env_value = 'hello'

    @staticmethod
    def validate(runtime_env_dict: dict) -> str:
        if False:
            i = 10
            return i + 15
        return None

    def modify_context(self, uris: List[str], plugin_config_dict: dict, ctx: RuntimeEnvContext, logger: logging.Logger) -> None:
        if False:
            for i in range(10):
                print('nop')
        if PRIORITY_TEST_ENV_VAR_NAME in ctx.env_vars:
            raise RuntimeError(f'Env var {PRIORITY_TEST_ENV_VAR_NAME} has been set to {ctx.env_vars[PRIORITY_TEST_ENV_VAR_NAME]}.')
        ctx.env_vars[PRIORITY_TEST_ENV_VAR_NAME] = PriorityTestPlugin2.env_value
priority_test_plugin_config_without_priority = [{'class': PRIORITY_TEST_PLUGIN1_CLASS_PATH}, {'class': PRIORITY_TEST_PLUGIN2_CLASS_PATH}]
priority_test_plugin_config = [{'class': PRIORITY_TEST_PLUGIN1_CLASS_PATH, 'priority': 1}, {'class': PRIORITY_TEST_PLUGIN2_CLASS_PATH, 'priority': 0}]
priority_test_plugin_bad_config = [{'class': PRIORITY_TEST_PLUGIN1_CLASS_PATH, 'priority': 0, 'tag': 'bad'}, {'class': PRIORITY_TEST_PLUGIN2_CLASS_PATH, 'priority': 1}]

@pytest.mark.parametrize('set_runtime_env_plugins', [json.dumps(priority_test_plugin_config_without_priority), json.dumps(priority_test_plugin_config), json.dumps(priority_test_plugin_bad_config)], indirect=True)
def test_plugin_priority(set_runtime_env_plugins, ray_start_regular):
    if False:
        while True:
            i = 10
    config = set_runtime_env_plugins
    (_, tmp_file_path) = tempfile.mkstemp()

    @ray.remote
    def f():
        if False:
            return 10
        import os
        return os.environ.get(PRIORITY_TEST_ENV_VAR_NAME)
    if 'bad' in config:
        with pytest.raises(RuntimeEnvSetupError, match='has been set'):
            value = ray.get(f.options(runtime_env={PRIORITY_TEST_PLUGIN1_NAME: {}, PRIORITY_TEST_PLUGIN2_NAME: {}}).remote())
    else:
        value = ray.get(f.options(runtime_env={PRIORITY_TEST_PLUGIN1_NAME: {}, PRIORITY_TEST_PLUGIN2_NAME: {}}).remote())
        assert value is not None
        assert value == 'hello world'

def test_unexpected_field_warning(shutdown_only):
    if False:
        i = 10
        return i + 15
    "Test that an unexpected runtime_env field doesn't error."
    ray.init(runtime_env={'unexpected_field': 'value'})

    @ray.remote
    def f():
        if False:
            for i in range(10):
                print('nop')
        return True
    assert ray.get(f.remote())
    session_dir = ray._private.worker.global_worker.node.address_info['session_dir']
    log_path = Path(session_dir) / 'logs'
    wait_for_condition(lambda : any(('unexpected_field is not recognized' in open(f).read() for f in log_path.glob('runtime_env_setup*.log'))))
URI_CACHING_TEST_PLUGIN_CLASS_PATH = 'ray.tests.test_runtime_env_plugin.UriCachingTestPlugin'
URI_CACHING_TEST_PLUGIN_NAME = 'UriCachingTestPlugin'
URI_CACHING_TEST_DIR = Path(tempfile.gettempdir()) / 'runtime_env_uri_caching_test'
uri_caching_test_file_path = URI_CACHING_TEST_DIR / 'uri_caching_test_file.json'
URI_CACHING_TEST_DIR.mkdir(parents=True, exist_ok=True)
uri_caching_test_file_path.write_text('{}')

def get_plugin_usage_data():
    if False:
        for i in range(10):
            print('nop')
    with open(uri_caching_test_file_path, 'r') as f:
        data = json.loads(f.read())
        return data

class UriCachingTestPlugin(RuntimeEnvPlugin):
    """A plugin that fakes taking up local disk space when creating its environment.

    This plugin is used to test that the URI caching is working correctly.
    Example:
        runtime_env = {"UriCachingTestPlugin": {"uri": "file:///a", "size_bytes": 10}}
    """
    name = URI_CACHING_TEST_PLUGIN_NAME

    def __init__(self):
        if False:
            return 10
        self.uris_to_sizes = {}
        self.modify_context_call_count = 0
        self.create_call_count = 0

    def write_plugin_usage_data(self) -> None:
        if False:
            print('Hello World!')
        with open(uri_caching_test_file_path, 'w') as f:
            data = {'uris_to_sizes': self.uris_to_sizes, 'modify_context_call_count': self.modify_context_call_count, 'create_call_count': self.create_call_count}
            f.write(json.dumps(data))

    def get_uris(self, runtime_env: 'RuntimeEnv') -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        return [runtime_env[self.name]['uri']]

    async def create(self, uri, runtime_env: 'RuntimeEnv', context: RuntimeEnvContext, logger: logging.Logger) -> float:
        self.create_call_count += 1
        created_size_bytes = runtime_env[self.name]['size_bytes']
        self.uris_to_sizes[uri] = created_size_bytes
        self.write_plugin_usage_data()
        return created_size_bytes

    def modify_context(self, uris: List[str], runtime_env: 'RuntimeEnv', context: RuntimeEnvContext, logger: logging.Logger) -> None:
        if False:
            i = 10
            return i + 15
        self.modify_context_call_count += 1
        self.write_plugin_usage_data()

    def delete_uri(self, uri: str, logger: logging.Logger) -> int:
        if False:
            for i in range(10):
                print('nop')
        size = self.uris_to_sizes.pop(uri)
        self.write_plugin_usage_data()
        return size

@pytest.fixture(scope='class')
def uri_cache_size_100_gb():
    if False:
        for i in range(10):
            print('nop')
    var = f'RAY_RUNTIME_ENV_{URI_CACHING_TEST_PLUGIN_NAME}_CACHE_SIZE_GB'.upper()
    with mock.patch.dict(os.environ, {var: '100'}):
        print('Set URI cache size for UriCachingTestPlugin to 100 GB')
        yield

def gb_to_bytes(size_gb: int) -> int:
    if False:
        while True:
            i = 10
    return size_gb * 1024 * 1024 * 1024

class TestGC:

    @pytest.mark.parametrize('set_runtime_env_plugins', [json.dumps([{'class': URI_CACHING_TEST_PLUGIN_CLASS_PATH}])], indirect=True)
    def test_uri_caching(self, set_runtime_env_plugins, start_cluster, uri_cache_size_100_gb):
        if False:
            for i in range(10):
                print('nop')
        (cluster, address) = start_cluster
        ray.init(address=address)

        def reinit():
            if False:
                return 10
            ray.shutdown()
            time.sleep(5)
            ray.init(address=address)

        @ray.remote
        def f():
            if False:
                i = 10
                return i + 15
            return True
        ref1 = f.options(runtime_env={URI_CACHING_TEST_PLUGIN_NAME: {'uri': 'file:///tmp/test_uri_1', 'size_bytes': gb_to_bytes(50)}}).remote()
        ray.get(ref1)
        print(get_plugin_usage_data())
        wait_for_condition(lambda : get_plugin_usage_data() == {'uris_to_sizes': {'file:///tmp/test_uri_1': gb_to_bytes(50)}, 'modify_context_call_count': 1, 'create_call_count': 1})
        reinit()
        ref2 = f.options(runtime_env={URI_CACHING_TEST_PLUGIN_NAME: {'uri': 'file:///tmp/test_uri_2', 'size_bytes': gb_to_bytes(51)}}).remote()
        ray.get(ref2)
        wait_for_condition(lambda : get_plugin_usage_data() == {'uris_to_sizes': {'file:///tmp/test_uri_2': gb_to_bytes(51)}, 'modify_context_call_count': 2, 'create_call_count': 2})
        reinit()
        ref3 = f.options(runtime_env={URI_CACHING_TEST_PLUGIN_NAME: {'uri': 'file:///tmp/test_uri_2', 'size_bytes': gb_to_bytes(51)}}).remote()
        ray.get(ref3)
        wait_for_condition(lambda : get_plugin_usage_data() == {'uris_to_sizes': {'file:///tmp/test_uri_2': gb_to_bytes(51)}, 'modify_context_call_count': 3, 'create_call_count': 2})
        reinit()
        ref4 = f.options(runtime_env={URI_CACHING_TEST_PLUGIN_NAME: {'uri': 'file:///tmp/test_uri_3', 'size_bytes': gb_to_bytes(10)}}).remote()
        ray.get(ref4)
        wait_for_condition(lambda : get_plugin_usage_data() == {'uris_to_sizes': {'file:///tmp/test_uri_2': gb_to_bytes(51), 'file:///tmp/test_uri_3': gb_to_bytes(10)}, 'modify_context_call_count': 4, 'create_call_count': 3})
if __name__ == '__main__':
    import sys
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))