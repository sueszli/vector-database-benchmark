import json
import os
import pathlib
import sys
import time
from dataclasses import asdict
from pathlib import Path
import requests
import pytest
from jsonschema import validate
from http.server import BaseHTTPRequestHandler, HTTPServer
import ray
import ray._private.usage.usage_constants as usage_constants
import ray._private.usage.usage_lib as ray_usage_lib
from ray._private.test_utils import format_web_url, run_string_as_driver, wait_for_condition, wait_until_server_available
from ray._private.usage.usage_lib import ClusterConfigToReport, UsageStatsEnabledness
from ray.autoscaler._private.cli_logger import cli_logger
from ray.util.placement_group import placement_group
schema = {'$schema': 'http://json-schema.org/draft-07/schema#', 'type': 'object', 'properties': {'schema_version': {'type': 'string'}, 'source': {'type': 'string'}, 'session_id': {'type': 'string'}, 'ray_version': {'type': 'string'}, 'git_commit': {'type': 'string'}, 'os': {'type': 'string'}, 'python_version': {'type': 'string'}, 'collect_timestamp_ms': {'type': 'integer'}, 'session_start_timestamp_ms': {'type': 'integer'}, 'cloud_provider': {'type': ['null', 'string']}, 'min_workers': {'type': ['null', 'integer']}, 'max_workers': {'type': ['null', 'integer']}, 'head_node_instance_type': {'type': ['null', 'string']}, 'libc_version': {'type': ['null', 'string']}, 'worker_node_instance_types': {'type': ['null', 'array'], 'items': {'type': 'string'}}, 'total_num_cpus': {'type': ['null', 'integer']}, 'total_num_gpus': {'type': ['null', 'integer']}, 'total_memory_gb': {'type': ['null', 'number']}, 'total_object_store_memory_gb': {'type': ['null', 'number']}, 'library_usages': {'type': ['null', 'array'], 'items': {'type': 'string'}}, 'total_success': {'type': 'integer'}, 'total_failed': {'type': 'integer'}, 'seq_number': {'type': 'integer'}, 'extra_usage_tags': {'type': ['null', 'object']}, 'total_num_nodes': {'type': ['null', 'integer']}, 'total_num_running_jobs': {'type': ['null', 'integer']}}, 'additionalProperties': False}

def file_exists(temp_dir: Path):
    if False:
        print('Hello World!')
    for path in temp_dir.iterdir():
        if usage_constants.USAGE_STATS_FILE in str(path):
            return True
    return False

def read_file(temp_dir: Path, column: str):
    if False:
        return 10
    usage_stats_file = temp_dir / usage_constants.USAGE_STATS_FILE
    with usage_stats_file.open() as f:
        result = json.load(f)
        return result[column]

def print_dashboard_log():
    if False:
        return 10
    session_dir = ray._private.worker.global_worker.node.address_info['session_dir']
    session_path = Path(session_dir)
    log_dir_path = session_path / 'logs'
    paths = list(log_dir_path.iterdir())
    contents = None
    for path in paths:
        if 'dashboard.log' in str(path):
            with open(str(path), 'r') as f:
                contents = f.readlines()
    from pprint import pprint
    pprint(contents)

@pytest.fixture
def gcs_storage_type():
    if False:
        for i in range(10):
            print('nop')
    storage = 'redis' if os.environ.get('RAY_REDIS_ADDRESS') else 'memory'
    yield storage

@pytest.fixture
def reset_usage_stats():
    if False:
        for i in range(10):
            print('nop')
    yield
    ray.experimental.internal_kv._internal_kv_reset()
    ray_usage_lib._recorded_library_usages.clear()
    ray_usage_lib._recorded_extra_usage_tags.clear()

@pytest.fixture
def reset_ray_version_commit():
    if False:
        while True:
            i = 10
    saved_ray_version = ray.__version__
    saved_ray_commit = ray.__commit__
    yield
    ray.__version__ = saved_ray_version
    ray.__commit__ = saved_ray_commit

@pytest.mark.parametrize('ray_client', [True, False])
def test_get_extra_usage_tags_to_report(monkeypatch, call_ray_start, reset_usage_stats, ray_client, gcs_storage_type):
    if False:
        while True:
            i = 10
    if os.environ.get('RAY_MINIMAL') == '1' and ray_client:
        pytest.skip("Skipping due to we don't have ray client in minimal.")
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_EXTRA_TAGS', 'key=val;key2=val2')
        result = ray_usage_lib.get_extra_usage_tags_to_report(ray.experimental.internal_kv.internal_kv_get_gcs_client())
        assert result['key'] == 'val'
        assert result['key2'] == 'val2'
        m.setenv('RAY_USAGE_STATS_EXTRA_TAGS', 'key=val;key2=val2;')
        result = ray_usage_lib.get_extra_usage_tags_to_report(ray.experimental.internal_kv.internal_kv_get_gcs_client())
        assert result['key'] == 'val'
        assert result['key2'] == 'val2'
        m.delenv('RAY_USAGE_STATS_EXTRA_TAGS')
        result = ray_usage_lib.get_extra_usage_tags_to_report(ray.experimental.internal_kv.internal_kv_get_gcs_client())
        assert result == {}
        m.setenv('RAY_USAGE_STATS_EXTRA_TAGS', 'key=val,key2=val2')
        result = ray_usage_lib.get_extra_usage_tags_to_report(ray.experimental.internal_kv.internal_kv_get_gcs_client())
        assert result == {}
        m.setenv('RAY_USAGE_STATS_EXTRA_TAGS', 'key=v=al,key2=val2')
        result = ray_usage_lib.get_extra_usage_tags_to_report(ray.experimental.internal_kv.internal_kv_get_gcs_client())
        assert result == {}
        address = call_ray_start
        ray.init(address=address)
        m.setenv('RAY_USAGE_STATS_EXTRA_TAGS', 'key=val')
        driver = '\nimport ray\nimport ray._private.usage.usage_lib as ray_usage_lib\n\nray_usage_lib.record_extra_usage_tag(ray_usage_lib.TagKey._TEST1, "val1")\nray.init(address="{}")\nray_usage_lib.record_extra_usage_tag(ray_usage_lib.TagKey._TEST2, "val2")\n'.format('ray://127.0.0.1:10001' if ray_client else address)
        run_string_as_driver(driver)
        wait_for_condition(lambda : ray_usage_lib.get_extra_usage_tags_to_report(ray.experimental.internal_kv.internal_kv_get_gcs_client()) == {'key': 'val', '_test1': 'val1', '_test2': 'val2', 'actor_num_created': '0', 'pg_num_created': '0', 'num_actor_creation_tasks': '0', 'num_actor_tasks': '0', 'num_normal_tasks': '0', 'num_drivers': '2', 'gcs_storage': gcs_storage_type, 'dashboard_used': 'False'}, timeout=10)
        ray_usage_lib.record_extra_usage_tag(ray_usage_lib.TagKey._TEST2, 'val3')
        wait_for_condition(lambda : ray_usage_lib.get_extra_usage_tags_to_report(ray.experimental.internal_kv.internal_kv_get_gcs_client()) == {'key': 'val', '_test1': 'val1', '_test2': 'val3', 'actor_num_created': '0', 'pg_num_created': '0', 'num_actor_creation_tasks': '0', 'num_actor_tasks': '0', 'num_normal_tasks': '0', 'num_drivers': '2', 'gcs_storage': gcs_storage_type, 'dashboard_used': 'False'}, timeout=10)

@pytest.mark.skipif(sys.platform != 'linux' and sys.platform != 'linux2', reason='memory monitor only on linux currently')
def test_worker_crash_increment_stats():
    if False:
        i = 10
        return i + 15

    @ray.remote
    def crasher():
        if False:
            while True:
                i = 10
        exit(1)

    @ray.remote
    def oomer():
        if False:
            i = 10
            return i + 15
        mem = []
        while True:
            mem.append([0] * 1000000000)
    with ray.init() as ctx:
        with pytest.raises(ray.exceptions.WorkerCrashedError):
            ray.get(crasher.options(max_retries=1).remote())
        with pytest.raises(ray.exceptions.OutOfMemoryError):
            ray.get(oomer.options(max_retries=0).remote())
        gcs_client = ray._raylet.GcsClient(address=ctx.address_info['gcs_address'])
        wait_for_condition(lambda : 'worker_crash_system_error' in ray_usage_lib.get_extra_usage_tags_to_report(gcs_client), timeout=4)
        result = ray_usage_lib.get_extra_usage_tags_to_report(gcs_client)
        assert 'worker_crash_system_error' in result
        assert result['worker_crash_system_error'] == '2'
        assert 'worker_crash_oom' in result
        assert result['worker_crash_oom'] == '1'

def test_actor_stats(reset_usage_stats):
    if False:
        return 10

    @ray.remote
    class Actor:

        def foo(self):
            if False:
                print('Hello World!')
            pass
    with ray.init(_system_config={'metrics_report_interval_ms': 1000}) as ctx:
        gcs_client = ray._raylet.GcsClient(address=ctx.address_info['gcs_address'])
        actor = Actor.remote()
        wait_for_condition(lambda : ray_usage_lib.get_extra_usage_tags_to_report(gcs_client).get('actor_num_created') == '1' and ray_usage_lib.get_extra_usage_tags_to_report(gcs_client).get('num_actor_creation_tasks') == '1', timeout=10)
        actor = Actor.remote()
        wait_for_condition(lambda : ray_usage_lib.get_extra_usage_tags_to_report(gcs_client).get('actor_num_created') == '2' and ray_usage_lib.get_extra_usage_tags_to_report(gcs_client).get('num_actor_creation_tasks') == '2' and (ray_usage_lib.get_extra_usage_tags_to_report(gcs_client).get('num_actor_tasks') == '0'), timeout=10)
        ray.get(actor.foo.remote())
        wait_for_condition(lambda : ray_usage_lib.get_extra_usage_tags_to_report(gcs_client).get('actor_num_created') == '2' and ray_usage_lib.get_extra_usage_tags_to_report(gcs_client).get('num_actor_creation_tasks') == '2' and (ray_usage_lib.get_extra_usage_tags_to_report(gcs_client).get('num_actor_tasks') == '1'), timeout=10)
        del actor

def test_pg_stats(reset_usage_stats):
    if False:
        while True:
            i = 10
    with ray.init(num_cpus=3, _system_config={'metrics_report_interval_ms': 1000}) as ctx:
        gcs_client = ray._raylet.GcsClient(address=ctx.address_info['gcs_address'])
        pg = placement_group([{'CPU': 1}], strategy='STRICT_PACK')
        ray.get(pg.ready())
        wait_for_condition(lambda : ray_usage_lib.get_extra_usage_tags_to_report(gcs_client).get('pg_num_created') == '1', timeout=5)
        pg1 = placement_group([{'CPU': 1}], strategy='STRICT_PACK')
        ray.get(pg1.ready())
        wait_for_condition(lambda : ray_usage_lib.get_extra_usage_tags_to_report(gcs_client).get('pg_num_created') == '2', timeout=5)

def test_task_stats(reset_usage_stats):
    if False:
        return 10

    @ray.remote
    def foo():
        if False:
            for i in range(10):
                print('nop')
        pass
    with ray.init(_system_config={'metrics_report_interval_ms': 1000}) as ctx:
        gcs_client = ray._raylet.GcsClient(address=ctx.address_info['gcs_address'])
        wait_for_condition(lambda : ray_usage_lib.get_extra_usage_tags_to_report(gcs_client).get('num_normal_tasks') == '0' and ray_usage_lib.get_extra_usage_tags_to_report(gcs_client).get('num_drivers') == '1', timeout=10)
        ray.get(foo.remote())
        wait_for_condition(lambda : ray_usage_lib.get_extra_usage_tags_to_report(gcs_client).get('num_normal_tasks') == '1', timeout=10)
        ray.get(foo.remote())
        wait_for_condition(lambda : ray_usage_lib.get_extra_usage_tags_to_report(gcs_client).get('num_normal_tasks') == '2' and ray_usage_lib.get_extra_usage_tags_to_report(gcs_client).get('num_drivers') == '1', timeout=10)

def test_usage_stats_enabledness(monkeypatch, tmp_path, reset_usage_stats):
    if False:
        return 10
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', '1')
        assert ray_usage_lib._usage_stats_enabledness() is UsageStatsEnabledness.ENABLED_EXPLICITLY
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', '0')
        assert ray_usage_lib._usage_stats_enabledness() is UsageStatsEnabledness.DISABLED_EXPLICITLY
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', 'xxx')
        with pytest.raises(ValueError):
            ray_usage_lib._usage_stats_enabledness()
    with monkeypatch.context() as m:
        m.delenv('RAY_USAGE_STATS_ENABLED', raising=False)
        tmp_usage_stats_config_path = tmp_path / 'config.json'
        monkeypatch.setenv('RAY_USAGE_STATS_CONFIG_PATH', str(tmp_usage_stats_config_path))
        tmp_usage_stats_config_path.write_text('{"usage_stats": true}')
        assert ray_usage_lib._usage_stats_enabledness() is UsageStatsEnabledness.ENABLED_EXPLICITLY
        tmp_usage_stats_config_path.write_text('{"usage_stats": false}')
        assert ray_usage_lib._usage_stats_enabledness() is UsageStatsEnabledness.DISABLED_EXPLICITLY
        tmp_usage_stats_config_path.write_text('{"usage_stats": "xxx"}')
        with pytest.raises(ValueError):
            ray_usage_lib._usage_stats_enabledness()
        tmp_usage_stats_config_path.write_text('')
        assert ray_usage_lib._usage_stats_enabledness() is UsageStatsEnabledness.ENABLED_BY_DEFAULT
        tmp_usage_stats_config_path.unlink()
        assert ray_usage_lib._usage_stats_enabledness() is UsageStatsEnabledness.ENABLED_BY_DEFAULT

def test_set_usage_stats_enabled_via_config(monkeypatch, tmp_path, reset_usage_stats):
    if False:
        while True:
            i = 10
    tmp_usage_stats_config_path = tmp_path / 'config1.json'
    monkeypatch.setenv('RAY_USAGE_STATS_CONFIG_PATH', str(tmp_usage_stats_config_path))
    ray_usage_lib.set_usage_stats_enabled_via_config(True)
    assert '{"usage_stats": true}' == tmp_usage_stats_config_path.read_text()
    ray_usage_lib.set_usage_stats_enabled_via_config(False)
    assert '{"usage_stats": false}' == tmp_usage_stats_config_path.read_text()
    tmp_usage_stats_config_path.write_text('"xxx"')
    ray_usage_lib.set_usage_stats_enabled_via_config(True)
    assert '{"usage_stats": true}' == tmp_usage_stats_config_path.read_text()
    tmp_usage_stats_config_path.unlink()
    os.makedirs(os.path.dirname(tmp_usage_stats_config_path / 'xxx.txt'), exist_ok=True)
    with pytest.raises(Exception, match='Failed to enable usage stats.*'):
        ray_usage_lib.set_usage_stats_enabled_via_config(True)

@pytest.fixture
def clear_loggers():
    if False:
        return 10
    'Remove handlers from all loggers'
    yield
    import logging
    loggers = [logging.getLogger()] + list(logging.Logger.manager.loggerDict.values())
    for logger in loggers:
        handlers = getattr(logger, 'handlers', [])
        for handler in handlers:
            logger.removeHandler(handler)

def test_usage_stats_prompt(monkeypatch, capsys, tmp_path, reset_usage_stats, shutdown_only, clear_loggers, reset_ray_version_commit):
    if False:
        print('Hello World!')
    '\n    Test usage stats prompt is shown in the proper cases.\n    '
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', '1')
        m.setenv('RAY_USAGE_STATS_PROMPT_ENABLED', '0')
        ray_usage_lib.show_usage_stats_prompt(cli=True)
        captured = capsys.readouterr()
        assert usage_constants.USAGE_STATS_ENABLED_FOR_CLI_MESSAGE not in captured.out
        assert usage_constants.USAGE_STATS_ENABLED_FOR_CLI_MESSAGE not in captured.err
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', '1')
        m.setenv('RAY_USAGE_STATS_PROMPT_ENABLED', '0')
        ray_usage_lib.show_usage_stats_prompt(cli=False)
        captured = capsys.readouterr()
        assert usage_constants.USAGE_STATS_ENABLED_FOR_RAY_INIT_MESSAGE not in captured.out
        assert usage_constants.USAGE_STATS_ENABLED_FOR_RAY_INIT_MESSAGE not in captured.err
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', '0')
        ray_usage_lib.show_usage_stats_prompt(cli=True)
        captured = capsys.readouterr()
        assert usage_constants.USAGE_STATS_DISABLED_MESSAGE in captured.out
    with monkeypatch.context() as m:
        m.delenv('RAY_USAGE_STATS_ENABLED', raising=False)
        tmp_usage_stats_config_path = tmp_path / 'config1.json'
        m.setenv('RAY_USAGE_STATS_CONFIG_PATH', str(tmp_usage_stats_config_path))
        ray_usage_lib.show_usage_stats_prompt(cli=True)
        captured = capsys.readouterr()
        assert usage_constants.USAGE_STATS_ENABLED_BY_DEFAULT_FOR_CLI_MESSAGE in captured.out
    with monkeypatch.context() as m:
        if sys.platform != 'win32':
            m.delenv('RAY_USAGE_STATS_ENABLED', raising=False)
            saved_interactive = cli_logger.interactive
            saved_stdin = sys.stdin
            tmp_usage_stats_config_path = tmp_path / 'config2.json'
            m.setenv('RAY_USAGE_STATS_CONFIG_PATH', str(tmp_usage_stats_config_path))
            cli_logger.interactive = True
            (r_pipe, w_pipe) = os.pipe()
            sys.stdin = open(r_pipe)
            os.write(w_pipe, b'y\n')
            ray_usage_lib.show_usage_stats_prompt(cli=True)
            captured = capsys.readouterr()
            assert usage_constants.USAGE_STATS_CONFIRMATION_MESSAGE in captured.out
            assert usage_constants.USAGE_STATS_ENABLED_FOR_CLI_MESSAGE in captured.out
            cli_logger.interactive = saved_interactive
            sys.stdin = saved_stdin
    with monkeypatch.context() as m:
        if sys.platform != 'win32':
            m.delenv('RAY_USAGE_STATS_ENABLED', raising=False)
            saved_interactive = cli_logger.interactive
            saved_stdin = sys.stdin
            tmp_usage_stats_config_path = tmp_path / 'config3.json'
            m.setenv('RAY_USAGE_STATS_CONFIG_PATH', str(tmp_usage_stats_config_path))
            cli_logger.interactive = True
            (r_pipe, w_pipe) = os.pipe()
            sys.stdin = open(r_pipe)
            os.write(w_pipe, b'n\n')
            ray_usage_lib.show_usage_stats_prompt(cli=True)
            captured = capsys.readouterr()
            assert usage_constants.USAGE_STATS_CONFIRMATION_MESSAGE in captured.out
            assert usage_constants.USAGE_STATS_DISABLED_MESSAGE in captured.out
            cli_logger.interactive = saved_interactive
            sys.stdin = saved_stdin
    with monkeypatch.context() as m:
        m.delenv('RAY_USAGE_STATS_ENABLED', raising=False)
        saved_interactive = cli_logger.interactive
        saved_stdin = sys.stdin
        tmp_usage_stats_config_path = tmp_path / 'config4.json'
        m.setenv('RAY_USAGE_STATS_CONFIG_PATH', str(tmp_usage_stats_config_path))
        cli_logger.interactive = True
        (r_pipe, w_pipe) = os.pipe()
        sys.stdin = open(r_pipe)
        ray_usage_lib.show_usage_stats_prompt(cli=True)
        captured = capsys.readouterr()
        assert usage_constants.USAGE_STATS_CONFIRMATION_MESSAGE in captured.out
        assert usage_constants.USAGE_STATS_ENABLED_FOR_CLI_MESSAGE in captured.out
        cli_logger.interactive = saved_interactive
        sys.stdin = saved_stdin
    with monkeypatch.context() as m:
        m.delenv('RAY_USAGE_STATS_ENABLED', raising=False)
        tmp_usage_stats_config_path = tmp_path / 'config5.json'
        m.setenv('RAY_USAGE_STATS_CONFIG_PATH', str(tmp_usage_stats_config_path))
        ray.__version__ = '2.0.0'
        ray.__commit__ = 'xyzf'
        ray.init()
        ray.shutdown()
        captured = capsys.readouterr()
        assert usage_constants.USAGE_STATS_ENABLED_BY_DEFAULT_FOR_RAY_INIT_MESSAGE not in captured.out
        assert usage_constants.USAGE_STATS_ENABLED_FOR_RAY_INIT_MESSAGE not in captured.out
    with monkeypatch.context() as m:
        m.delenv('RAY_USAGE_STATS_ENABLED', raising=False)
        tmp_usage_stats_config_path = tmp_path / 'config6.json'
        m.setenv('RAY_USAGE_STATS_CONFIG_PATH', str(tmp_usage_stats_config_path))
        ray.__version__ = '2.0.0.dev0'
        ray.__commit__ = 'xyzf'
        ray.init()
        ray.shutdown()
        captured = capsys.readouterr()
        assert usage_constants.USAGE_STATS_ENABLED_BY_DEFAULT_FOR_RAY_INIT_MESSAGE in captured.out
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', '0')
        ray.__version__ = '2.0.0.dev0'
        ray.__commit__ = 'xyzf'
        ray.init()
        ray.shutdown()
        captured = capsys.readouterr()
        assert usage_constants.USAGE_STATS_DISABLED_MESSAGE in captured.out
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', '1')
        ray.__version__ = '2.0.0.dev0'
        ray.__commit__ = 'xyzf'
        ray.init()
        ray.shutdown()
        captured = capsys.readouterr()
        assert usage_constants.USAGE_STATS_ENABLED_FOR_RAY_INIT_MESSAGE in captured.out

def test_is_nightly_wheel(reset_ray_version_commit):
    if False:
        print('Hello World!')
    ray.__version__ = '2.0.0'
    ray.__commit__ = 'xyz'
    assert not ray_usage_lib.is_nightly_wheel()
    ray.__version__ = '2.0.0dev0'
    ray.__commit__ = '{{RAY_COMMIT_SHA}}'
    assert not ray_usage_lib.is_nightly_wheel()
    ray.__version__ = '2.0.0dev0'
    ray.__commit__ = 'xyz'
    assert ray_usage_lib.is_nightly_wheel()

def test_usage_lib_cluster_metadata_generation(monkeypatch, ray_start_cluster, reset_usage_stats):
    if False:
        print('Hello World!')
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', '1')
        m.setenv('RAY_USAGE_STATS_REPORT_URL', 'http://127.0.0.1:8000')
        cluster = ray_start_cluster
        cluster.add_node(num_cpus=0)
        ray.init(address=cluster.address)
        '\n        Test metadata stored is equivalent to `_generate_cluster_metadata`.\n        '
        meta = ray_usage_lib._generate_cluster_metadata()
        cluster_metadata = ray_usage_lib.get_cluster_metadata(ray.experimental.internal_kv.internal_kv_get_gcs_client())
        assert meta.pop('session_id')
        assert meta.pop('session_start_timestamp_ms')
        assert cluster_metadata.pop('session_id')
        assert cluster_metadata.pop('session_start_timestamp_ms')
        assert meta == cluster_metadata
        '\n        Make sure put & get works properly.\n        '
        cluster_metadata = ray_usage_lib.put_cluster_metadata(ray.experimental.internal_kv.internal_kv_get_gcs_client())
        assert cluster_metadata == ray_usage_lib.get_cluster_metadata(ray.experimental.internal_kv.internal_kv_get_gcs_client())

@pytest.mark.skipif(os.environ.get('RAY_MINIMAL') == '1', reason='This test is not supposed to work for minimal installation.')
def test_usage_stats_enabled_endpoint(monkeypatch, ray_start_cluster, reset_usage_stats):
    if False:
        return 10
    import requests
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', '0')
        m.setenv('RAY_USAGE_STATS_PROMPT_ENABLED', '0')
        cluster = ray_start_cluster
        cluster.add_node(num_cpus=0)
        context = ray.init(address=cluster.address)
        webui_url = context['webui_url']
        assert wait_until_server_available(webui_url)
        webui_url = format_web_url(webui_url)
        response = requests.get(f'{webui_url}/usage_stats_enabled')
        assert response.status_code == 200
        assert response.json()['result'] is True
        assert response.json()['data']['usageStatsEnabled'] is False
        assert response.json()['data']['usageStatsPromptEnabled'] is False

@pytest.mark.skipif(os.environ.get('RAY_MINIMAL') == '1', reason='This test is not supposed to work for minimal installation since we import serve.')
@pytest.mark.parametrize('ray_client', [True, False])
def test_library_usages(call_ray_start, reset_usage_stats, ray_client):
    if False:
        return 10
    from ray.job_submission import JobSubmissionClient
    address = call_ray_start
    ray.init(address=address)
    driver = '\nimport ray\nimport ray._private.usage.usage_lib as ray_usage_lib\n\nray_usage_lib.record_library_usage("pre_init")\nray.init(address="{}")\n\nray_usage_lib.record_library_usage("post_init")\nray.workflow.init()\nray.data.range(10)\nfrom ray import serve\n\nserve.start()\nserve.shutdown()\n\nclass Actor:\n    def get_actor_metadata(self):\n        return "metadata"\n\nfrom ray.util.actor_group import ActorGroup\nactor_group = ActorGroup(Actor)\n\nactor_pool = ray.util.actor_pool.ActorPool([])\n\nfrom ray.util.multiprocessing import Pool\npool = Pool()\n\nfrom ray.util.queue import Queue\nqueue = Queue()\n\nimport joblib\nfrom ray.util.joblib import register_ray\nregister_ray()\nwith joblib.parallel_backend("ray"):\n    pass\n'.format('ray://127.0.0.1:10001' if ray_client else address)
    run_string_as_driver(driver)
    if sys.platform != 'win32':
        job_submission_client = JobSubmissionClient('http://127.0.0.1:8265')
        job_id = job_submission_client.submit_job(entrypoint='ls')
        wait_for_condition(lambda : job_submission_client.get_job_status(job_id) == ray.job_submission.JobStatus.SUCCEEDED)
    library_usages = ray_usage_lib.get_library_usages_to_report(ray.experimental.internal_kv.internal_kv_get_gcs_client())
    expected = {'pre_init', 'post_init', 'dataset', 'workflow', 'serve', 'util.ActorGroup', 'util.ActorPool', 'util.multiprocessing.Pool', 'util.Queue', 'util.joblib'}
    if sys.platform != 'win32':
        expected.add('job_submission')
    if ray_client:
        expected.add('client')
    assert set(library_usages) == expected

def test_usage_lib_cluster_metadata_generation_usage_disabled(monkeypatch, shutdown_only, reset_usage_stats):
    if False:
        print('Hello World!')
    '\n    Make sure only version information is generated when usage stats are not enabled.\n    '
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', '0')
        meta = ray_usage_lib._generate_cluster_metadata()
        assert 'ray_version' in meta
        assert 'python_version' in meta
        assert len(meta) == 2

def test_usage_lib_get_total_num_running_jobs_to_report(ray_start_cluster, reset_usage_stats):
    if False:
        return 10
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=1)
    gcs_client = ray._raylet.GcsClient(address=cluster.gcs_address)
    assert ray_usage_lib.get_total_num_running_jobs_to_report(gcs_client) == 0
    ray.init(address=cluster.address)
    assert ray_usage_lib.get_total_num_running_jobs_to_report(gcs_client) == 1
    ray.shutdown()
    ray.init(address=cluster.address)
    assert ray_usage_lib.get_total_num_running_jobs_to_report(gcs_client) == 1
    ray.shutdown()

def test_usage_lib_get_total_num_nodes_to_report(ray_start_cluster, reset_usage_stats):
    if False:
        print('Hello World!')
    cluster = ray_start_cluster
    cluster.add_node(num_cpus=1)
    ray.init(address=cluster.address)
    worker_node = cluster.add_node(num_cpus=2)
    assert ray_usage_lib.get_total_num_nodes_to_report(ray.experimental.internal_kv.internal_kv_get_gcs_client()) == 2
    cluster.remove_node(worker_node)
    assert ray_usage_lib.get_total_num_nodes_to_report(ray.experimental.internal_kv.internal_kv_get_gcs_client()) == 1

def test_usage_lib_get_cluster_status_to_report(shutdown_only, reset_usage_stats):
    if False:
        while True:
            i = 10
    ray.init(num_cpus=3, num_gpus=1, object_store_memory=2 ** 30)
    wait_for_condition(lambda : ray_usage_lib.get_cluster_status_to_report(ray.experimental.internal_kv.internal_kv_get_gcs_client()).total_num_cpus == 3, timeout=10)
    cluster_status_to_report = ray_usage_lib.get_cluster_status_to_report(ray.experimental.internal_kv.internal_kv_get_gcs_client())
    assert cluster_status_to_report.total_num_cpus == 3
    assert cluster_status_to_report.total_num_gpus == 1
    assert cluster_status_to_report.total_memory_gb > 0
    assert cluster_status_to_report.total_object_store_memory_gb == 1.0

def test_usage_lib_get_cluster_config_to_report(monkeypatch, tmp_path, reset_usage_stats):
    if False:
        return 10
    cluster_config_file_path = tmp_path / 'ray_bootstrap_config.yaml'
    ' Test minimal cluster config'
    cluster_config_file_path.write_text('\ncluster_name: minimal\nmax_workers: 1\nprovider:\n    type: aws\n    region: us-west-2\n    availability_zone: us-west-2a\n')
    cluster_config_to_report = ray_usage_lib.get_cluster_config_to_report(cluster_config_file_path)
    assert cluster_config_to_report.cloud_provider == 'aws'
    assert cluster_config_to_report.min_workers is None
    assert cluster_config_to_report.max_workers == 1
    assert cluster_config_to_report.head_node_instance_type is None
    assert cluster_config_to_report.worker_node_instance_types is None
    cluster_config_file_path.write_text('\ncluster_name: full\nmin_workers: 1\nprovider:\n    type: gcp\nhead_node_type: head_node\navailable_node_types:\n    head_node:\n        node_config:\n            InstanceType: m5.large\n        min_workers: 0\n        max_workers: 0\n    aws_worker_node:\n        node_config:\n            InstanceType: m3.large\n        min_workers: 0\n        max_workers: 0\n    azure_worker_node:\n        node_config:\n            azure_arm_parameters:\n                vmSize: Standard_D2s_v3\n    gcp_worker_node:\n        node_config:\n            machineType: n1-standard-2\n')
    cluster_config_to_report = ray_usage_lib.get_cluster_config_to_report(cluster_config_file_path)
    assert cluster_config_to_report.cloud_provider == 'gcp'
    assert cluster_config_to_report.min_workers == 1
    assert cluster_config_to_report.max_workers is None
    assert cluster_config_to_report.head_node_instance_type == 'm5.large'
    assert set(cluster_config_to_report.worker_node_instance_types) == {'m3.large', 'Standard_D2s_v3', 'n1-standard-2'}
    cluster_config_file_path.write_text('\ncluster_name: full\nhead_node_type: head_node\navailable_node_types:\n    worker_node_1:\n        node_config:\n            ImageId: xyz\n    worker_node_2:\n        resources: {}\n    worker_node_3:\n        node_config:\n            InstanceType: m5.large\n')
    cluster_config_to_report = ray_usage_lib.get_cluster_config_to_report(cluster_config_file_path)
    assert cluster_config_to_report.cloud_provider is None
    assert cluster_config_to_report.min_workers is None
    assert cluster_config_to_report.max_workers is None
    assert cluster_config_to_report.head_node_instance_type is None
    assert cluster_config_to_report.worker_node_instance_types == ['m5.large']
    cluster_config_file_path.write_text('[invalid')
    cluster_config_to_report = ray_usage_lib.get_cluster_config_to_report(cluster_config_file_path)
    assert cluster_config_to_report == ClusterConfigToReport()
    cluster_config_to_report = ray_usage_lib.get_cluster_config_to_report(tmp_path / 'does_not_exist.yaml')
    assert cluster_config_to_report == ClusterConfigToReport()
    monkeypatch.setenv('KUBERNETES_SERVICE_HOST', 'localhost')
    cluster_config_to_report = ray_usage_lib.get_cluster_config_to_report(tmp_path / 'does_not_exist.yaml')
    assert cluster_config_to_report.cloud_provider == 'kubernetes'
    assert cluster_config_to_report.min_workers is None
    assert cluster_config_to_report.max_workers is None
    assert cluster_config_to_report.head_node_instance_type is None
    assert cluster_config_to_report.worker_node_instance_types is None
    monkeypatch.setenv('RAY_USAGE_STATS_KUBERAY_IN_USE', '1')
    cluster_config_to_report = ray_usage_lib.get_cluster_config_to_report(tmp_path / 'does_not_exist.yaml')
    assert cluster_config_to_report.cloud_provider == 'kuberay'

@pytest.mark.skipif(sys.version_info >= (3, 11, 0), reason='Currently not passing for Python 3.11')
def test_usage_lib_report_data(monkeypatch, ray_start_cluster, tmp_path, reset_usage_stats):
    if False:
        while True:
            i = 10
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', '1')
        m.setenv('RAY_USAGE_STATS_REPORT_URL', 'http://127.0.0.1:8000')
        cluster = ray_start_cluster
        cluster.add_node(num_cpus=0)
        ray.init(address=cluster.address)
        '\n        Make sure the generated data is following the schema.\n        '
        cluster_config_file_path = tmp_path / 'ray_bootstrap_config.yaml'
        cluster_config_file_path.write_text('\ncluster_name: minimal\nmax_workers: 1\nprovider:\n    type: aws\n    region: us-west-2\n    availability_zone: us-west-2a\n')
        cluster_config_to_report = ray_usage_lib.get_cluster_config_to_report(cluster_config_file_path)
        d = ray_usage_lib.generate_report_data(cluster_config_to_report, 2, 2, 2, ray.worker.global_worker.gcs_client.address)
        validate(instance=asdict(d), schema=schema)
        '\n        Make sure writing to a file works as expected\n        '
        client = ray_usage_lib.UsageReportClient()
        temp_dir = Path(tmp_path)
        client.write_usage_data(d, temp_dir)
        wait_for_condition(lambda : file_exists(temp_dir))
        '\n        Make sure report usage data works as expected\n        '

        class UsageStatsServer(BaseHTTPRequestHandler):
            expected_data = None

            def do_POST(self):
                if False:
                    i = 10
                    return i + 15
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                if json.loads(post_data) == self.expected_data:
                    self.send_response(200)
                else:
                    self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()

        @ray.remote(num_cpus=0)
        def run_usage_stats_server(expected_data):
            if False:
                print('Hello World!')
            UsageStatsServer.expected_data = expected_data
            server = HTTPServer(('127.0.0.1', 8000), UsageStatsServer)
            server.serve_forever()
        run_usage_stats_server.remote(asdict(d))
        wait_for_condition(lambda : client.report_usage_data('http://127.0.0.1:8000', d), timeout=30)

@pytest.mark.skipif(sys.version_info >= (3, 11, 0), reason='Currently not passing for Python 3.11')
def test_usage_report_e2e(monkeypatch, ray_start_cluster, tmp_path, reset_usage_stats, gcs_storage_type):
    if False:
        while True:
            i = 10
    '\n    Test usage report works e2e with env vars.\n    '
    cluster_config_file_path = tmp_path / 'ray_bootstrap_config.yaml'
    cluster_config_file_path.write_text('\ncluster_name: minimal\nmax_workers: 1\nprovider:\n    type: aws\n    region: us-west-2\n    availability_zone: us-west-2a\n')
    with monkeypatch.context() as m:
        m.setenv('HOME', str(tmp_path))
        m.setenv('RAY_USAGE_STATS_ENABLED', '1')
        m.setenv('RAY_USAGE_STATS_REPORT_URL', 'http://127.0.0.1:8000')
        m.setenv('RAY_USAGE_STATS_REPORT_INTERVAL_S', '1')
        m.setenv('RAY_USAGE_STATS_EXTRA_TAGS', 'extra_k1=extra_v1')
        cluster = ray_start_cluster
        cluster.add_node(num_cpus=3)
        if os.environ.get('RAY_MINIMAL') != '1':
            from ray import train
            from ray.rllib.algorithms.ppo import PPO
        ray_usage_lib.record_extra_usage_tag(ray_usage_lib.TagKey._TEST1, 'extra_v2')
        ray.init(address=cluster.address)
        ray_usage_lib.record_extra_usage_tag(ray_usage_lib.TagKey._TEST2, 'extra_v3')
        if os.environ.get('RAY_MINIMAL') != '1':
            from ray import tune

            def objective(*args):
                if False:
                    i = 10
                    return i + 15
                pass
            tuner = tune.Tuner(objective)
            tuner.fit()

        @ray.remote(num_cpus=0)
        class StatusReporter:

            def __init__(self):
                if False:
                    return 10
                self.reported = 0
                self.payload = None

            def report_payload(self, payload):
                if False:
                    print('Hello World!')
                self.payload = payload

            def reported(self):
                if False:
                    return 10
                self.reported += 1

            def get(self):
                if False:
                    while True:
                        i = 10
                return self.reported

            def get_payload(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.payload
        reporter = StatusReporter.remote()

        class UsageStatsServer(BaseHTTPRequestHandler):
            reporter = None

            def do_POST(self):
                if False:
                    return 10
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                self.reporter.reported.remote()
                self.reporter.report_payload.remote(json.loads(post_data))
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()

        @ray.remote(num_cpus=0)
        def run_usage_stats_server(reporter):
            if False:
                for i in range(10):
                    print('nop')
            UsageStatsServer.reporter = reporter
            server = HTTPServer(('127.0.0.1', 8000), UsageStatsServer)
            server.serve_forever()
        run_usage_stats_server.remote(reporter)
        '\n        Verify the usage stats are reported to the server.\n        '
        print('Verifying usage stats report.')
        try:
            wait_for_condition(lambda : ray.get(reporter.get.remote()) > 5, timeout=30)
        except Exception:
            print_dashboard_log()
            raise
        payload = ray.get(reporter.get_payload.remote())
        (ray_version, python_version) = ray._private.utils.compute_version_info()
        assert payload['ray_version'] == ray_version
        assert payload['python_version'] == python_version
        assert payload['schema_version'] == '0.1'
        assert payload['os'] == sys.platform
        if sys.platform != 'linux':
            payload['libc_version'] is None
        else:
            import platform
            assert payload['libc_version'] == f'{platform.libc_ver()[0]}:{platform.libc_ver()[1]}'
        assert payload['source'] == 'OSS'
        assert payload['cloud_provider'] == 'aws'
        assert payload['min_workers'] is None
        assert payload['max_workers'] == 1
        assert payload['head_node_instance_type'] is None
        assert payload['worker_node_instance_types'] is None
        assert payload['total_num_cpus'] == 3
        assert payload['total_num_gpus'] is None
        assert payload['total_memory_gb'] > 0
        assert payload['total_object_store_memory_gb'] > 0
        assert int(payload['extra_usage_tags']['actor_num_created']) >= 0
        assert int(payload['extra_usage_tags']['pg_num_created']) >= 0
        assert int(payload['extra_usage_tags']['num_actor_creation_tasks']) >= 0
        assert int(payload['extra_usage_tags']['num_actor_tasks']) >= 0
        assert int(payload['extra_usage_tags']['num_normal_tasks']) >= 0
        assert int(payload['extra_usage_tags']['num_drivers']) >= 0
        payload['extra_usage_tags']['actor_num_created'] = '0'
        payload['extra_usage_tags']['pg_num_created'] = '0'
        payload['extra_usage_tags']['num_actor_creation_tasks'] = '0'
        payload['extra_usage_tags']['num_actor_tasks'] = '0'
        payload['extra_usage_tags']['num_normal_tasks'] = '0'
        payload['extra_usage_tags']['num_drivers'] = '0'
        expected_payload = {'extra_k1': 'extra_v1', '_test1': 'extra_v2', '_test2': 'extra_v3', 'dashboard_metrics_grafana_enabled': 'False', 'dashboard_metrics_prometheus_enabled': 'False', 'actor_num_created': '0', 'pg_num_created': '0', 'num_actor_creation_tasks': '0', 'num_actor_tasks': '0', 'num_normal_tasks': '0', 'num_drivers': '0', 'gcs_storage': gcs_storage_type, 'dashboard_used': 'False'}
        if os.environ.get('RAY_MINIMAL') != '1':
            expected_payload['tune_scheduler'] = 'FIFOScheduler'
            expected_payload['tune_searcher'] = 'BasicVariantGenerator'
            expected_payload['air_entrypoint'] = 'Tuner.fit'
            expected_payload['air_storage_configuration'] = 'local'
        assert payload['extra_usage_tags'] == expected_payload
        assert payload['total_num_nodes'] == 1
        assert payload['total_num_running_jobs'] == 1
        if os.environ.get('RAY_MINIMAL') == '1':
            assert set(payload['library_usages']) == set()
        else:
            assert set(payload['library_usages']) == {'rllib', 'train', 'tune'}
        validate(instance=payload, schema=schema)
        '\n        Verify the usage_stats.json is updated.\n        '
        print('Verifying usage stats write.')
        global_node = ray._private.worker._global_node
        temp_dir = pathlib.Path(global_node.get_session_dir_path())
        wait_for_condition(lambda : file_exists(temp_dir), timeout=30)
        timestamp_old = read_file(temp_dir, 'usage_stats')['collect_timestamp_ms']
        success_old = read_file(temp_dir, 'usage_stats')['total_success']
        wait_for_condition(lambda : timestamp_old < read_file(temp_dir, 'usage_stats')['collect_timestamp_ms'])
        wait_for_condition(lambda : success_old < read_file(temp_dir, 'usage_stats')['total_success'])
        assert read_file(temp_dir, 'success')

def test_first_usage_report_delayed(monkeypatch, ray_start_cluster, reset_usage_stats):
    if False:
        for i in range(10):
            print('nop')
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', '1')
        m.setenv('RAY_USAGE_STATS_REPORT_URL', 'http://127.0.0.1:8000')
        m.setenv('RAY_USAGE_STATS_REPORT_INTERVAL_S', '10')
        cluster = ray_start_cluster
        cluster.add_node(num_cpus=0)
        ray.init(address=cluster.address)
        time.sleep(5)
        session_dir = ray._private.worker.global_worker.node.address_info['session_dir']
        session_path = Path(session_dir)
        assert not (session_path / usage_constants.USAGE_STATS_FILE).exists()
        time.sleep(10)
        assert (session_path / usage_constants.USAGE_STATS_FILE).exists()

def test_usage_report_disabled(monkeypatch, ray_start_cluster, reset_usage_stats):
    if False:
        i = 10
        return i + 15
    '\n    Make sure usage report module is disabled when the env var is not set.\n    It also verifies that the failure message is not printed (note that\n    the invalid report url is given as an env var).\n    '
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', '0')
        m.setenv('RAY_USAGE_STATS_REPORT_URL', 'http://127.0.0.1:8000')
        m.setenv('RAY_USAGE_STATS_REPORT_INTERVAL_S', '1')
        cluster = ray_start_cluster
        cluster.add_node(num_cpus=0)
        ray.init(address=cluster.address)
        time.sleep(5)
        session_dir = ray._private.worker.global_worker.node.address_info['session_dir']
        session_path = Path(session_dir)
        log_dir_path = session_path / 'logs'
        paths = list(log_dir_path.iterdir())
        contents = None
        for path in paths:
            if 'dashboard.log' in str(path):
                with open(str(path), 'r') as f:
                    contents = f.readlines()
                break
        assert contents is not None
        assert any(['Usage reporting is disabled' in c for c in contents])
        assert all(['Failed to report usage stats' not in c for c in contents])

def test_usage_file_error_message(monkeypatch, ray_start_cluster, reset_usage_stats):
    if False:
        i = 10
        return i + 15
    '\n    Make sure the usage report file is generated with a proper\n    error message when the report is failed.\n    '
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', '1')
        m.setenv('RAY_USAGE_STATS_REPORT_URL', 'http://127.0.0.1:8000')
        m.setenv('RAY_USAGE_STATS_REPORT_INTERVAL_S', '1')
        cluster = ray_start_cluster
        cluster.add_node(num_cpus=0)
        ray.init(address=cluster.address)
        global_node = ray._private.worker._global_node
        temp_dir = pathlib.Path(global_node.get_session_dir_path())
        try:
            wait_for_condition(lambda : file_exists(temp_dir), timeout=30)
        except Exception:
            print_dashboard_log()
            raise
        error_message = read_file(temp_dir, 'error')
        failure_old = read_file(temp_dir, 'usage_stats')['total_failed']
        report_success = read_file(temp_dir, 'success')
        assert "HTTPConnectionPool(host='127.0.0.1', port=8000): Max retries exceeded with url:" in error_message
        assert not report_success
        try:
            wait_for_condition(lambda : failure_old < read_file(temp_dir, 'usage_stats')['total_failed'])
        except Exception:
            print_dashboard_log()
            read_file(temp_dir, 'usage_stats')['total_failed']
            raise
        assert read_file(temp_dir, 'usage_stats')['total_success'] == 0

def test_lib_used_from_driver(monkeypatch, ray_start_cluster, reset_usage_stats):
    if False:
        return 10
    '\n    Test library usage is correctly reported when they are imported from\n    a driver.\n    '
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', '1')
        m.setenv('RAY_USAGE_STATS_REPORT_URL', 'http://127.0.0.1:8000/usage')
        m.setenv('RAY_USAGE_STATS_REPORT_INTERVAL_S', '1')
        cluster = ray_start_cluster
        cluster.add_node(num_cpus=3)
        ray.init(address=cluster.address)
        script = '\nimport ray\nimport os\nif os.environ.get("RAY_MINIMAL") != "1":\n    from ray import train  # noqa: F401\n    from ray.rllib.algorithms.ppo import PPO  # noqa: F401\n\nray.init(address="{addr}")\n\nif os.environ.get("RAY_MINIMAL") != "1":\n    from ray import tune  # noqa: F401\n    def objective(*args):\n        pass\n\n    tune.run(objective)\n'
        run_string_as_driver(script.format(addr=cluster.address))
        '\n        Verify the usage_stats.json is updated.\n        '
        print('Verifying lib usage report.')
        global_node = ray.worker._global_node
        temp_dir = pathlib.Path(global_node.get_session_dir_path())
        wait_for_condition(lambda : file_exists(temp_dir), timeout=30)

        def verify():
            if False:
                print('Hello World!')
            lib_usages = read_file(temp_dir, 'usage_stats')['library_usages']
            print(lib_usages)
            if os.environ.get('RAY_MINIMAL') == '1':
                return set(lib_usages) == set()
            else:
                return set(lib_usages) == {'rllib', 'train', 'tune'}
        wait_for_condition(verify)

@pytest.mark.skipif(os.environ.get('RAY_MINIMAL') == '1', reason='This test is not supposed to work for minimal installation.')
@pytest.mark.skipif(sys.version_info >= (3, 11, 0), reason='Currently not passing for Python 3.11')
def test_lib_used_from_workers(monkeypatch, ray_start_cluster, reset_usage_stats):
    if False:
        print('Hello World!')
    '\n    Test library usage is correctly reported when they are imported from\n    workers.\n    '
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', '1')
        m.setenv('RAY_USAGE_STATS_REPORT_URL', 'http://127.0.0.1:8000/usage')
        m.setenv('RAY_USAGE_STATS_REPORT_INTERVAL_S', '1')
        cluster = ray_start_cluster
        cluster.add_node(num_cpus=3)
        ray_usage_lib._recorded_library_usages.clear()
        ray.init(address=cluster.address)

        @ray.remote
        class ActorWithLibImport:

            def __init__(self):
                if False:
                    return 10
                from ray import train
                from ray.rllib.algorithms.ppo import PPO

            def ready(self):
                if False:
                    while True:
                        i = 10
                from ray import tune

                def objective(*args):
                    if False:
                        for i in range(10):
                            print('nop')
                    pass
                tune.run(objective)
        a = ActorWithLibImport.remote()
        ray.get(a.ready.remote())
        '\n        Verify the usage_stats.json contains the lib usage.\n        '
        global_node = ray.worker._global_node
        temp_dir = pathlib.Path(global_node.get_session_dir_path())
        wait_for_condition(lambda : file_exists(temp_dir), timeout=30)

        def verify():
            if False:
                i = 10
                return i + 15
            lib_usages = read_file(temp_dir, 'usage_stats')['library_usages']
            return set(lib_usages) == {'tune', 'rllib', 'train'}
        wait_for_condition(verify)

def test_usage_stats_tags(monkeypatch, ray_start_cluster, reset_usage_stats, gcs_storage_type):
    if False:
        while True:
            i = 10
    '\n    Test usage tags are correctly reported.\n    '
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', '1')
        m.setenv('RAY_USAGE_STATS_REPORT_URL', 'http://127.0.0.1:8000/usage')
        m.setenv('RAY_USAGE_STATS_REPORT_INTERVAL_S', '1')
        m.setenv('RAY_USAGE_STATS_EXTRA_TAGS', 'key=val;key2=val2')
        cluster = ray_start_cluster
        cluster.add_node(num_cpus=3)
        cluster.add_node(num_cpus=3)
        context = ray.init(address=cluster.address)
        '\n        Verify the usage_stats.json contains the lib usage.\n        '
        temp_dir = pathlib.Path(context.address_info['session_dir'])
        wait_for_condition(lambda : file_exists(temp_dir), timeout=30)

        def verify():
            if False:
                for i in range(10):
                    print('nop')
            tags = read_file(temp_dir, 'usage_stats')['extra_usage_tags']
            num_nodes = read_file(temp_dir, 'usage_stats')['total_num_nodes']
            assert tags == {'key': 'val', 'key2': 'val2', 'dashboard_metrics_grafana_enabled': 'False', 'dashboard_metrics_prometheus_enabled': 'False', 'gcs_storage': gcs_storage_type, 'dashboard_used': 'False', 'actor_num_created': '0', 'pg_num_created': '0', 'num_actor_creation_tasks': '0', 'num_actor_tasks': '0', 'num_normal_tasks': '0', 'num_drivers': '1'}
            assert num_nodes == 2
            return True
        wait_for_condition(verify)

def test_usage_stats_gcs_query_failure(monkeypatch, ray_start_cluster, reset_usage_stats):
    if False:
        return 10
    'Test None data is reported when the GCS query is failed.'
    with monkeypatch.context() as m:
        m.setenv('RAY_testing_asio_delay_us', 'NodeInfoGcsService.grpc_server.GetAllNodeInfo=2000000:2000000')
        cluster = ray_start_cluster
        cluster.add_node(num_cpus=3)
        ray.init(address=cluster.address)
        assert ray_usage_lib.get_total_num_nodes_to_report(ray.experimental.internal_kv.internal_kv_get_gcs_client(), timeout=1) is None

def test_usages_stats_available_when_dashboard_not_included(monkeypatch, ray_start_cluster, reset_usage_stats):
    if False:
        i = 10
        return i + 15
    '\n    Test library usage is correctly reported when they are imported from\n    workers.\n    '
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', '1')
        m.setenv('RAY_USAGE_STATS_REPORT_URL', 'http://127.0.0.1:8000/usage')
        m.setenv('RAY_USAGE_STATS_REPORT_INTERVAL_S', '1')
        cluster = ray_start_cluster
        cluster.add_node(num_cpus=1, include_dashboard=False)
        ray.init(address=cluster.address)
        '\n        Verify the usage_stats.json contains the lib usage.\n        '
        temp_dir = pathlib.Path(cluster.head_node.get_session_dir_path())
        wait_for_condition(lambda : file_exists(temp_dir), timeout=30)

        def verify():
            if False:
                i = 10
                return i + 15
            return read_file(temp_dir, 'usage_stats')['seq_number'] > 2
        wait_for_condition(verify)

def test_usages_stats_dashboard(monkeypatch, ray_start_cluster, reset_usage_stats):
    if False:
        return 10
    '\n    Test dashboard usage metrics are correctly reported.\n    This is tested on both minimal / non minimal ray.\n    '
    with monkeypatch.context() as m:
        m.setenv('RAY_USAGE_STATS_ENABLED', '1')
        m.setenv('RAY_USAGE_STATS_REPORT_URL', 'http://127.0.0.1:8000/usage')
        m.setenv('RAY_USAGE_STATS_REPORT_INTERVAL_S', '1')
        cluster = ray_start_cluster
        cluster.add_node(num_cpus=0)
        addr = ray.init(address=cluster.address)
        '\n        Verify the usage_stats.json contains the lib usage.\n        '
        temp_dir = pathlib.Path(ray._private.worker._global_node.get_session_dir_path())
        webui_url = format_web_url(addr['webui_url'])
        wait_for_condition(lambda : file_exists(temp_dir), timeout=30)

        def verify_dashboard_not_used():
            if False:
                while True:
                    i = 10
            dashboard_used = read_file(temp_dir, 'usage_stats')['extra_usage_tags']['dashboard_used']
            return dashboard_used == 'False'
        wait_for_condition(verify_dashboard_not_used)
        if os.environ.get('RAY_MINIMAL') == '1':
            return
        resp = requests.get(webui_url)
        resp.raise_for_status()

        def verify_dashboard_used():
            if False:
                i = 10
                return i + 15
            dashboard_used = read_file(temp_dir, 'usage_stats')['extra_usage_tags']['dashboard_used']
            if os.environ.get('RAY_MINIMAL') == '1':
                return dashboard_used == 'False'
            else:
                return dashboard_used == 'True'
        wait_for_condition(verify_dashboard_used)
if __name__ == '__main__':
    if os.environ.get('PARALLEL_CI'):
        sys.exit(pytest.main(['-n', 'auto', '--boxed', '-vs', __file__]))
    else:
        sys.exit(pytest.main(['-sv', __file__]))