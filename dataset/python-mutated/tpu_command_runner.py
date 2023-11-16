"""Command runners specific to TPU VM pods.

TPU VM pods may contain multiple hosts, each including attached TPU chips and
associated internal/external IP addresses.

To support TPU VM pods, we represent entire TPU pods as "Ray Nodes", meaning
that TPU pods will need to run the operations specified in `CommandRunnerInterface`
N times, where N denotes the number of hosts that comprise a TPU pod.

To maintain feature completeness, we simply wrap the existing `SSHCommandRunner` and
`DockerCommandRunner` and run them as batched calls.

"""
import copy
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, Dict, Optional
from ray._private import ray_constants
from ray.autoscaler._private.command_runner import DockerCommandRunner, SSHCommandRunner
from ray.autoscaler._private.gcp.node import GCPTPUNode
from ray.autoscaler.command_runner import CommandRunnerInterface
from ray.autoscaler.node_provider import NodeProvider

class TPUVMSSHCommandRunner(SSHCommandRunner):
    """An SSH command runner with overwritten IP address calls."""

    def __init__(self, internal_ip: str, external_ip: str, worker_id: int, accelerator_type: str, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._internal_ip = internal_ip
        self._external_ip = external_ip
        self._worker_id = worker_id
        self._accelerator_type = accelerator_type
        super().__init__(*args, **kwargs)

    def _get_node_ip(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        if self.use_internal_ip:
            return self._internal_ip
        else:
            return self._external_ip

    def run(self, cmd, timeout=120, exit_on_fail=False, port_forward=None, with_output=False, environment_variables: Dict[str, object]=None, run_env='auto', ssh_options_override_ssh_key='', shutdown_after_run=False):
        if False:
            for i in range(10):
                print('nop')
        'Override the SSH run for TPU VM pods.\n\n        Main functionality here we need to inject is to intercept the resources\n        provided by the node_provider TPU node type fillout.\n\n        node_provider will provide a resource "TPU-{TPU_POD_TYPE}-head" which:\n        1) allows application developers to target worker 0 of an arbitary TPU pod, and\n        2) signals to the autoscaler how to address the demand for more TPU pods.\n\n        Without this intercept, then all workers of a TPU pod will have the\n        "TPU-{TPU_POD_TYPE}-head" resource which will violate functionality (1)\n        above.\n\n        '
        if environment_variables:
            resources = environment_variables.get(ray_constants.RESOURCES_ENVIRONMENT_VARIABLE, None)
            if resources:
                if self._worker_id != 0:
                    tpu_pod_resource_type = f'TPU-{self._accelerator_type}-head'
                    if tpu_pod_resource_type in resources:
                        resources.pop(tpu_pod_resource_type, None)
                environment_variables[ray_constants.RESOURCES_ENVIRONMENT_VARIABLE] = resources
        super().run(cmd=cmd, timeout=timeout, exit_on_fail=exit_on_fail, port_forward=port_forward, with_output=with_output, environment_variables=environment_variables, run_env=run_env, ssh_options_override_ssh_key=ssh_options_override_ssh_key, shutdown_after_run=shutdown_after_run)

class TPUVMDockerCommandRunner(DockerCommandRunner):
    """A Docker command runner with overwritten IP addresses."""

    def __init__(self, docker_config: Dict[str, Any], internal_ip: str, external_ip: str, worker_id: int, accelerator_type: str, **common_args):
        if False:
            return 10
        super().__init__(docker_config=docker_config, **common_args)
        self.ssh_command_runner = TPUVMSSHCommandRunner(internal_ip=internal_ip, external_ip=external_ip, worker_id=worker_id, accelerator_type=accelerator_type, **common_args)

class TPUCommandRunner(CommandRunnerInterface):
    """A TPU pod command runner."""

    def __init__(self, instance: GCPTPUNode, log_prefix: str, node_id: str, auth_config: Dict[str, Any], provider: NodeProvider, cluster_name: str, process_runner: ModuleType, use_internal_ip: bool, docker_config: Optional[Dict[str, Any]]=None):
        if False:
            for i in range(10):
                print('nop')

        def create_command_runner(worker_id: int, accelerator_type: str, internal_ip: str, external_ip: str) -> CommandRunnerInterface:
            if False:
                return 10
            'Returns the correct base command runner.'
            common_args = {'internal_ip': internal_ip, 'external_ip': external_ip, 'worker_id': worker_id, 'accelerator_type': accelerator_type, 'log_prefix': '[tpu_worker_{}] '.format(worker_id) + log_prefix, 'node_id': node_id, 'provider': provider, 'auth_config': auth_config, 'cluster_name': cluster_name, 'process_runner': process_runner, 'use_internal_ip': use_internal_ip}
            if docker_config and docker_config['container_name'] != '':
                return TPUVMDockerCommandRunner(docker_config=docker_config, **common_args)
            else:
                return TPUVMSSHCommandRunner(**common_args)
        self._command_runners = []
        self._num_workers = instance.num_workers
        for i in range(self._num_workers):
            self._command_runners.append(create_command_runner(worker_id=i, accelerator_type=instance.get('acceleratorType'), internal_ip=instance.get_internal_ip(i), external_ip=instance.get_external_ip(i)))

    @property
    def num_connections(self) -> int:
        if False:
            i = 10
            return i + 15
        'Return the number of active connections allowed at a time.\n\n        We occasionally see issues where too many concurrent connections may lead to\n        failed SSH connections when there are too many TPU hosts.\n\n        We utilize this property to cap the maximum number of active connections\n        at a time until a proper fix is found.\n\n        '
        num_max_concurrent_active_connections = ray_constants.env_integer(ray_constants.RAY_TPU_MAX_CONCURRENT_CONNECTIONS_ENV_VAR, default=16)
        return min(self._num_workers, num_max_concurrent_active_connections)

    def run(self, cmd, timeout=120, exit_on_fail=False, port_forward=None, with_output=False, environment_variables: Dict[str, object]=None, run_env='auto', ssh_options_override_ssh_key='', shutdown_after_run=False) -> str:
        if False:
            for i in range(10):
                print('nop')
        with ThreadPoolExecutor(self.num_connections) as executor:
            results = executor.map(lambda i: self._command_runners[i].run(cmd=cmd, timeout=timeout, exit_on_fail=exit_on_fail, port_forward=port_forward, with_output=with_output, environment_variables=copy.deepcopy(environment_variables), run_env=run_env, ssh_options_override_ssh_key=ssh_options_override_ssh_key, shutdown_after_run=shutdown_after_run), range(self._num_workers))
        return list(results)[0]

    def run_rsync_up(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        with ThreadPoolExecutor(self.num_connections) as executor:
            executor.map(lambda i: self._command_runners[i].run_rsync_up(*args, **kwargs), range(self._num_workers))

    def run_rsync_down(self, *args, **kwargs) -> None:
        if False:
            print('Hello World!')
        'Rsync files down from the cluster node.\n\n        Args:\n            source: The (remote) source directory or file.\n            target: The (local) destination path.\n        '
        with ThreadPoolExecutor(self.num_connections) as executor:
            executor.map(lambda i: self._command_runners[i].run_rsync_down(*args, **kwargs), range(self._num_workers))

    def remote_shell_command_str(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return the command the user can use to open a shell.'
        return self._command_runners[0].remote_shell_command_str()

    def run_init(self, *args, **kwargs) -> Optional[bool]:
        if False:
            for i in range(10):
                print('nop')
        'Used to run extra initialization commands.\n\n        Args:\n            as_head: Run as head image or worker.\n            file_mounts: Files to copy to the head and worker nodes.\n            sync_run_yet: Whether sync has been run yet.\n\n        Returns:\n            optional: Whether initialization is necessary.\n        '
        with ThreadPoolExecutor(self.num_connections) as executor:
            results = executor.map(lambda i: self._command_runners[i].run_init(*args, **kwargs), range(self._num_workers))
        return any(results)