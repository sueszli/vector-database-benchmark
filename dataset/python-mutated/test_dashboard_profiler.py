import pytest
import subprocess
import os
import requests
import sys
import ray
from ray._private.test_utils import format_web_url, wait_until_server_available, wait_until_succeeded_without_exception

@pytest.mark.skipif(os.environ.get('RAY_MINIMAL') == '1', reason='This test is not supposed to work for minimal installation.')
@pytest.mark.skipif(sys.platform == 'win32', reason='No py-spy on Windows.')
@pytest.mark.skipif(sys.platform == 'darwin', reason='Fails on OSX: https://github.com/ray-project/ray/issues/30114')
@pytest.mark.parametrize('native', ['0', '1'])
def test_profiler_endpoints(ray_start_with_dashboard, native):
    if False:
        while True:
            i = 10
    subprocess.check_call(['py-spy', '--version'])
    assert wait_until_server_available(ray_start_with_dashboard['webui_url']) is True
    address_info = ray_start_with_dashboard
    webui_url = address_info['webui_url']
    webui_url = format_web_url(webui_url)

    @ray.remote
    class Actor:

        def getpid(self):
            if False:
                return 10
            return os.getpid()

        def do_stuff_infinite(self):
            if False:
                return 10
            while True:
                pass
    a = Actor.remote()
    pid = ray.get(a.getpid.remote())
    a.do_stuff_infinite.remote()

    def get_actor_stack():
        if False:
            for i in range(10):
                print('nop')
        url = f'{webui_url}/worker/traceback?pid={pid}&native={native}'
        print('GET URL', url)
        response = requests.get(url)
        print('STATUS CODE', response.status_code)
        print('HEADERS', response.headers)
        content = response.content.decode('utf-8')
        print('CONTENT', content)
        response.raise_for_status()
        assert 'text/plain' in response.headers['Content-Type'], response.headers
        assert 'do_stuff_infinite' in content, content
        if native == '1':
            assert 'ray::core::CoreWorker' in content, content
        else:
            assert 'ray::core::CoreWorker' not in content, content
    assert wait_until_succeeded_without_exception(get_actor_stack, (requests.RequestException, AssertionError), timeout_ms=20000, retry_interval_ms=1000)

    def get_actor_flamegraph():
        if False:
            print('Hello World!')
        response = requests.get(f'{webui_url}/worker/cpu_profile?pid={pid}&native={native}')
        response.raise_for_status()
        assert response.headers['Content-Type'] == 'image/svg+xml', response.headers
        content = response.content.decode('utf-8')
        print(content)
        assert '<!DOCTYPE svg' in content, content
        assert 'do_stuff_infinite' in content, content
        if native == '1':
            assert 'ray::core' in content, content
        else:
            assert 'ray::core' not in content, content
    assert wait_until_succeeded_without_exception(get_actor_flamegraph, (requests.RequestException, AssertionError), timeout_ms=20000, retry_interval_ms=1000)

@pytest.mark.skipif(os.environ.get('RAY_MINIMAL') == '1', reason='This test is not supposed to work for minimal installation.')
@pytest.mark.skipif(sys.platform == 'win32', reason='No py-spy on Windows.')
@pytest.mark.skipif(sys.platform == 'darwin', reason='Fails on OSX: https://github.com/ray-project/ray/issues/30114')
def test_profiler_failure_message(ray_start_with_dashboard):
    if False:
        return 10
    subprocess.check_call(['py-spy', '--version'])
    assert wait_until_server_available(ray_start_with_dashboard['webui_url']) is True
    address_info = ray_start_with_dashboard
    webui_url = address_info['webui_url']
    webui_url = format_web_url(webui_url)

    @ray.remote
    class Actor:

        def getpid(self):
            if False:
                while True:
                    i = 10
            return os.getpid()

        def do_stuff_infinite(self):
            if False:
                i = 10
                return i + 15
            while True:
                pass
    a = Actor.remote()
    pid = ray.get(a.getpid.remote())
    a.do_stuff_infinite.remote()

    def get_actor_stack():
        if False:
            for i in range(10):
                print('nop')
        response = requests.get(f'{webui_url}/worker/traceback?pid={pid}')
        response.raise_for_status()
        content = response.content.decode('utf-8')
        print('CONTENT', content)
        assert 'do_stuff_infinite' in content, content
    assert wait_until_succeeded_without_exception(get_actor_stack, (requests.RequestException, AssertionError), timeout_ms=20000, retry_interval_ms=1000)
    response = requests.get(f'{webui_url}/worker/traceback?pid=1234567')
    content = response.content.decode('utf-8')
    print(content)
    assert 'text/plain' in response.headers['Content-Type'], response.headers
    assert 'Failed to execute' in content, content
    response = requests.get(f'{webui_url}/worker/cpu_profile?pid=1234567')
    content = response.content.decode('utf-8')
    print(content)
    assert 'text/plain' in response.headers['Content-Type'], response.headers
    assert 'Failed to execute' in content, content
if __name__ == '__main__':
    sys.exit(pytest.main(['-v', '-s', __file__]))