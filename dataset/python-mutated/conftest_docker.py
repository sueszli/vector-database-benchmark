import time
import pytest
from pytest_docker_tools import container, fetch, network, volume
from pytest_docker_tools import wrappers

class Container(wrappers.Container):

    def ready(self):
        if False:
            i = 10
            return i + 15
        self._container.reload()
        if self.status == 'exited':
            from pytest_docker_tools.exceptions import ContainerFailed
            raise ContainerFailed(self, f'Container {self.name} has already exited before we noticed it was ready')
        if self.status != 'running':
            return False
        networks = self._container.attrs['NetworkSettings']['Networks']
        for (_, n) in networks.items():
            if not n['IPAddress']:
                return False
        if 'Ray runtime started' in super().logs():
            return True
        return False

    def client(self):
        if False:
            i = 10
            return i + 15
        from http.client import HTTPConnection
        port = self.ports['8000/tcp'][0]
        return HTTPConnection(f'localhost:{port}')

    def print_logs(self):
        if False:
            while True:
                i = 10
        for (name, content) in self.get_files('/tmp'):
            print(f'===== log start:  {name} ====')
            print(content.decode())
gcs_network = network(driver='bridge')
redis_image = fetch(repository='redis:latest')
redis = container(image='{redis_image.id}', network='{gcs_network.name}', command='redis-server --save 60 1 --loglevel warning')
head_node_vol = volume()
worker_node_vol = volume()
head_node_container_name = 'gcs' + str(int(time.time()))
head_node = container(image='ray_ci:v1', name=head_node_container_name, network='{gcs_network.name}', command=['ray', 'start', '--head', '--block', '--num-cpus', '0', '--node-manager-port', '9379'], volumes={'{head_node_vol.name}': {'bind': '/tmp', 'mode': 'rw'}}, environment={'RAY_REDIS_ADDRESS': '{redis.ips.primary}:6379', 'RAY_raylet_client_num_connect_attempts': '10', 'RAY_raylet_client_connect_timeout_milliseconds': '100'}, wrapper_class=Container, ports={'8000/tcp': None})
worker_node = container(image='ray_ci:v1', network='{gcs_network.name}', command=['ray', 'start', '--address', f'{head_node_container_name}:6379', '--block', '--node-manager-port', '9379'], volumes={'{worker_node_vol.name}': {'bind': '/tmp', 'mode': 'rw'}}, environment={'RAY_REDIS_ADDRESS': '{redis.ips.primary}:6379', 'RAY_raylet_client_num_connect_attempts': '10', 'RAY_raylet_client_connect_timeout_milliseconds': '100'}, wrapper_class=Container, ports={'8000/tcp': None})

@pytest.fixture
def docker_cluster(head_node, worker_node):
    if False:
        print('Hello World!')
    yield (head_node, worker_node)