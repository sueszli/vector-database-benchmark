from contextlib import contextmanager
import time
from typing import Any, Dict
import ray as real_ray
from ray.job_config import JobConfig
import ray.util.client.server.server as ray_client_server
from ray.util.client import ray
from ray._private.client_mode_hook import enable_client_mode, disable_client_hook

@contextmanager
def ray_start_client_server(metadata=None, ray_connect_handler=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    with ray_start_client_server_pair(metadata=metadata, ray_connect_handler=ray_connect_handler, **kwargs) as pair:
        (client, server) = pair
        yield client

@contextmanager
def ray_start_client_server_for_address(address):
    if False:
        for i in range(10):
            print('nop')
    '\n    Starts a Ray client server that initializes drivers at the specified address.\n    '

    def connect_handler(job_config: JobConfig=None, **ray_init_kwargs: Dict[str, Any]):
        if False:
            while True:
                i = 10
        import ray
        with disable_client_hook():
            if not ray.is_initialized():
                return ray.init(address, job_config=job_config, **ray_init_kwargs)
    with ray_start_client_server(ray_connect_handler=connect_handler) as ray:
        yield ray

@contextmanager
def ray_start_client_server_pair(metadata=None, ray_connect_handler=None, **kwargs):
    if False:
        i = 10
        return i + 15
    ray._inside_client_test = True
    with disable_client_hook():
        assert not ray.is_initialized()
    server = ray_client_server.serve('127.0.0.1:50051', ray_connect_handler=ray_connect_handler)
    ray.connect('127.0.0.1:50051', metadata=metadata, **kwargs)
    try:
        yield (ray, server)
    finally:
        ray._inside_client_test = False
        ray.disconnect()
        server.stop(0)
        del server
        start = time.monotonic()
        with disable_client_hook():
            while ray.is_initialized():
                time.sleep(1)
                if time.monotonic() - start > 30:
                    raise RuntimeError('Failed to terminate Ray')
        time.sleep(3)

@contextmanager
def ray_start_cluster_client_server_pair(address):
    if False:
        print('Hello World!')
    ray._inside_client_test = True

    def ray_connect_handler(job_config=None, **ray_init_kwargs):
        if False:
            print('Hello World!')
        real_ray.init(address=address)
    server = ray_client_server.serve('127.0.0.1:50051', ray_connect_handler=ray_connect_handler)
    ray.connect('127.0.0.1:50051')
    try:
        yield (ray, server)
    finally:
        ray._inside_client_test = False
        ray.disconnect()
        server.stop(0)

@contextmanager
def connect_to_client_or_not(connect_to_client: bool):
    if False:
        for i in range(10):
            print('nop')
    'Utility for running test logic with and without a Ray client connection.\n\n    If client_connect is True, will connect to Ray client in context.\n    If client_connect is False, does nothing.\n\n    How to use:\n    Given a test of the following form:\n\n    def test_<name>(args):\n        <initialize a ray cluster>\n        <use the ray cluster>\n\n    Modify the test to\n\n    @pytest.mark.parametrize("connect_to_client", [False, True])\n    def test_<name>(args, connect_to_client)\n    <initialize a ray cluster>\n    with connect_to_client_or_not(connect_to_client):\n        <use the ray cluster>\n\n    Parameterize the argument connect over True, False to run the test with and\n    without a Ray client connection.\n    '
    if connect_to_client:
        with ray_start_client_server(namespace=''), enable_client_mode():
            yield
    else:
        yield