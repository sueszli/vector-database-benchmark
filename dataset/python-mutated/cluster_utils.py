import copy
import json
import logging
import os
import subprocess
import tempfile
import time
from typing import Dict, Optional
import yaml
import ray
import ray._private.services
from ray._private import ray_constants
from ray._private.client_mode_hook import disable_client_hook
from ray._raylet import GcsClientOptions
from ray.util.annotations import DeveloperAPI
logger = logging.getLogger(__name__)
cluster_not_supported = os.name == 'nt'

@DeveloperAPI
class AutoscalingCluster:
    """Create a local autoscaling cluster for testing.

    See test_autoscaler_fake_multinode.py for an end-to-end example.
    """

    def __init__(self, head_resources: dict, worker_node_types: dict, **config_kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Create the cluster.\n\n        Args:\n            head_resources: resources of the head node, including CPU.\n            worker_node_types: autoscaler node types config for worker nodes.\n        '
        self._head_resources = head_resources
        self._config = self._generate_config(head_resources, worker_node_types, **config_kwargs)

    def _generate_config(self, head_resources, worker_node_types, **config_kwargs):
        if False:
            for i in range(10):
                print('nop')
        base_config = yaml.safe_load(open(os.path.join(os.path.dirname(ray.__file__), 'autoscaler/_private/fake_multi_node/example.yaml')))
        custom_config = copy.deepcopy(base_config)
        custom_config['available_node_types'] = worker_node_types
        custom_config['available_node_types']['ray.head.default'] = {'resources': head_resources, 'node_config': {}, 'max_workers': 0}
        custom_config.update(config_kwargs)
        return custom_config

    def start(self, _system_config=None, override_env: Optional[Dict]=None):
        if False:
            return 10
        'Start the cluster.\n\n        After this call returns, you can connect to the cluster with\n        ray.init("auto").\n        '
        subprocess.check_call(['ray', 'stop', '--force'])
        (_, fake_config) = tempfile.mkstemp()
        with open(fake_config, 'w') as f:
            f.write(json.dumps(self._config))
        cmd = ['ray', 'start', '--autoscaling-config={}'.format(fake_config), '--head']
        if 'CPU' in self._head_resources:
            cmd.append('--num-cpus={}'.format(self._head_resources.pop('CPU')))
        if 'GPU' in self._head_resources:
            cmd.append('--num-gpus={}'.format(self._head_resources.pop('GPU')))
        if 'object_store_memory' in self._head_resources:
            cmd.append('--object-store-memory={}'.format(self._head_resources.pop('object_store_memory')))
        if self._head_resources:
            cmd.append("--resources='{}'".format(json.dumps(self._head_resources)))
        if _system_config is not None:
            cmd.append('--system-config={}'.format(json.dumps(_system_config, separators=(',', ':'))))
        env = os.environ.copy()
        env.update({'AUTOSCALER_UPDATE_INTERVAL_S': '1', 'RAY_FAKE_CLUSTER': '1'})
        if override_env:
            env.update(override_env)
        subprocess.check_call(cmd, env=env)

    def shutdown(self):
        if False:
            i = 10
            return i + 15
        'Terminate the cluster.'
        subprocess.check_call(['ray', 'stop', '--force'])

@DeveloperAPI
class Cluster:

    def __init__(self, initialize_head: bool=False, connect: bool=False, head_node_args: dict=None, shutdown_at_exit: bool=True):
        if False:
            print('Hello World!')
        'Initializes all services of a Ray cluster.\n\n        Args:\n            initialize_head: Automatically start a Ray cluster\n                by initializing the head node. Defaults to False.\n            connect: If `initialize_head=True` and `connect=True`,\n                ray.init will be called with the address of this cluster\n                passed in.\n            head_node_args: Arguments to be passed into\n                `start_ray_head` via `self.add_node`.\n            shutdown_at_exit: If True, registers an exit hook\n                for shutting down all started processes.\n        '
        if cluster_not_supported:
            logger.warning('Ray cluster mode is currently experimental and untested on Windows. If you are using it and running into issues please file a report at https://github.com/ray-project/ray/issues.')
        self.head_node = None
        self.worker_nodes = set()
        self.redis_address = None
        self.connected = False
        self.global_state = ray._private.state.GlobalState()
        self._shutdown_at_exit = shutdown_at_exit
        if not initialize_head and connect:
            raise RuntimeError('Cannot connect to uninitialized cluster.')
        if initialize_head:
            head_node_args = head_node_args or {}
            self.add_node(**head_node_args)
            if connect:
                self.connect()

    @property
    def gcs_address(self):
        if False:
            i = 10
            return i + 15
        if self.head_node is None:
            return None
        return self.head_node.gcs_address

    @property
    def address(self):
        if False:
            while True:
                i = 10
        return self.gcs_address

    def connect(self, namespace=None):
        if False:
            print('Hello World!')
        'Connect the driver to the cluster.'
        assert self.address is not None
        assert not self.connected
        output_info = ray.init(namespace=namespace, ignore_reinit_error=True, address=self.address, _redis_password=self.redis_password)
        logger.info(output_info)
        self.connected = True

    def add_node(self, wait: bool=True, **node_args):
        if False:
            while True:
                i = 10
        'Adds a node to the local Ray Cluster.\n\n        All nodes are by default started with the following settings:\n            cleanup=True,\n            num_cpus=1,\n            object_store_memory=150 * 1024 * 1024  # 150 MiB\n\n        Args:\n            wait: Whether to wait until the node is alive.\n            node_args: Keyword arguments used in `start_ray_head` and\n                `start_ray_node`. Overrides defaults.\n\n        Returns:\n            Node object of the added Ray node.\n        '
        default_kwargs = {'num_cpus': 1, 'num_gpus': 0, 'object_store_memory': 150 * 1024 * 1024, 'min_worker_port': 0, 'max_worker_port': 0, 'dashboard_port': None}
        ray_params = ray._private.parameter.RayParams(**node_args)
        ray_params.update_if_absent(**default_kwargs)
        with disable_client_hook():
            if self.head_node is None:
                node = ray._private.node.Node(ray_params, head=True, shutdown_at_exit=self._shutdown_at_exit, spawn_reaper=self._shutdown_at_exit)
                self.head_node = node
                self.redis_address = self.head_node.redis_address
                self.redis_password = node_args.get('redis_password', ray_constants.REDIS_DEFAULT_PASSWORD)
                self.webui_url = self.head_node.webui_url
                gcs_options = GcsClientOptions.from_gcs_address(node.gcs_address)
                self.global_state._initialize_global_state(gcs_options)
                ray._private.utils.write_ray_address(self.head_node.gcs_address)
            else:
                ray_params.update_if_absent(redis_address=self.redis_address)
                ray_params.update_if_absent(gcs_address=self.gcs_address)
                ray_params.update_if_absent(include_log_monitor=False)
                ray_params.update_if_absent(node_manager_port=0)
                node = ray._private.node.Node(ray_params, head=False, shutdown_at_exit=self._shutdown_at_exit, spawn_reaper=self._shutdown_at_exit)
                self.worker_nodes.add(node)
            if wait:
                self._wait_for_node(node)
        return node

    def remove_node(self, node, allow_graceful=True):
        if False:
            return 10
        'Kills all processes associated with worker node.\n\n        Args:\n            node: Worker node of which all associated processes\n                will be removed.\n        '
        global_node = ray._private.worker._global_node
        if global_node is not None:
            if node._raylet_socket_name == global_node._raylet_socket_name:
                ray.shutdown()
                raise ValueError('Removing a node that is connected to this Ray client is not allowed because it will break the driver.You can use the get_other_node utility to avoid removinga node that the Ray client is connected.')
        if self.head_node == node:
            self.head_node.kill_all_processes(check_alive=False, allow_graceful=allow_graceful, wait=True)
            self.head_node = None
        else:
            node.kill_all_processes(check_alive=False, allow_graceful=allow_graceful, wait=True)
            self.worker_nodes.remove(node)
        assert not node.any_processes_alive(), 'There are zombie processes left over after killing.'

    def _wait_for_node(self, node, timeout: float=30):
        if False:
            while True:
                i = 10
        'Wait until this node has appeared in the client table.\n\n        Args:\n            node (ray._private.node.Node): The node to wait for.\n            timeout: The amount of time in seconds to wait before raising an\n                exception.\n\n        Raises:\n            TimeoutError: An exception is raised if the timeout expires before\n                the node appears in the client table.\n        '
        ray._private.services.wait_for_node(node.gcs_address, node.plasma_store_socket_name, timeout)

    def wait_for_nodes(self, timeout: float=30):
        if False:
            while True:
                i = 10
        'Waits for correct number of nodes to be registered.\n\n        This will wait until the number of live nodes in the client table\n        exactly matches the number of "add_node" calls minus the number of\n        "remove_node" calls that have been made on this cluster. This means\n        that if a node dies without "remove_node" having been called, this will\n        raise an exception.\n\n        Args:\n            timeout: The number of seconds to wait for nodes to join\n                before failing.\n\n        Raises:\n            TimeoutError: An exception is raised if we time out while waiting\n                for nodes to join.\n        '
        start_time = time.time()
        while time.time() - start_time < timeout:
            clients = self.global_state.node_table()
            live_clients = [client for client in clients if client['Alive']]
            expected = len(self.list_all_nodes())
            if len(live_clients) == expected:
                logger.debug('All nodes registered as expected.')
                return
            else:
                logger.debug(f'{len(live_clients)} nodes are currently registered, but we are expecting {expected}')
                time.sleep(0.1)
        raise TimeoutError('Timed out while waiting for nodes to join.')

    def list_all_nodes(self):
        if False:
            i = 10
            return i + 15
        'Lists all nodes.\n\n        TODO(rliaw): What is the desired behavior if a head node\n        dies before worker nodes die?\n\n        Returns:\n            List of all nodes, including the head node.\n        '
        nodes = list(self.worker_nodes)
        if self.head_node:
            nodes = [self.head_node] + nodes
        return nodes

    def remaining_processes_alive(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a bool indicating whether all processes are alive or not.\n\n        Note that this ignores processes that have been explicitly killed,\n        e.g., via a command like node.kill_raylet().\n\n        Returns:\n            True if all processes are alive and false otherwise.\n        '
        return all((node.remaining_processes_alive() for node in self.list_all_nodes()))

    def shutdown(self):
        if False:
            while True:
                i = 10
        'Removes all nodes.'
        all_nodes = list(self.worker_nodes)
        for node in all_nodes:
            self.remove_node(node)
        if self.head_node is not None:
            self.remove_node(self.head_node)
        ray.experimental.internal_kv._internal_kv_reset()
        ray._private.utils.reset_ray_address()