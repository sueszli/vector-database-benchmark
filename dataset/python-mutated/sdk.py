"""IMPORTANT: this is an experimental interface and not currently stable."""
import json
import os
import tempfile
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
from ray.autoscaler._private import commands
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.event_system import CreateClusterEvent
from ray.autoscaler._private.event_system import global_event_system
from ray.util.annotations import DeveloperAPI

@DeveloperAPI
def create_or_update_cluster(cluster_config: Union[dict, str], *, no_restart: bool=False, restart_only: bool=False, no_config_cache: bool=False) -> Dict[str, Any]:
    if False:
        while True:
            i = 10
    "Create or updates an autoscaling Ray cluster from a config json.\n\n    Args:\n        cluster_config (Union[str, dict]): Either the config dict of the\n            cluster, or a path pointing to a file containing the config.\n        no_restart: Whether to skip restarting Ray services during the\n            update. This avoids interrupting running jobs and can be used to\n            dynamically adjust autoscaler configuration.\n        restart_only: Whether to skip running setup commands and only\n            restart Ray. This cannot be used with 'no-restart'.\n        no_config_cache: Whether to disable the config cache and fully\n            resolve all environment settings from the Cloud provider again.\n    "
    with _as_config_file(cluster_config) as config_file:
        return commands.create_or_update_cluster(config_file=config_file, override_min_workers=None, override_max_workers=None, no_restart=no_restart, restart_only=restart_only, yes=True, override_cluster_name=None, no_config_cache=no_config_cache, redirect_command_output=None, use_login_shells=True)

@DeveloperAPI
def teardown_cluster(cluster_config: Union[dict, str], workers_only: bool=False, keep_min_workers: bool=False) -> None:
    if False:
        print('Hello World!')
    'Destroys all nodes of a Ray cluster described by a config json.\n\n    Args:\n        cluster_config (Union[str, dict]): Either the config dict of the\n            cluster, or a path pointing to a file containing the config.\n        workers_only: Whether to keep the head node running and only\n            teardown worker nodes.\n        keep_min_workers: Whether to keep min_workers (as specified\n            in the YAML) still running.\n    '
    with _as_config_file(cluster_config) as config_file:
        return commands.teardown_cluster(config_file=config_file, yes=True, workers_only=workers_only, override_cluster_name=None, keep_min_workers=keep_min_workers)

@DeveloperAPI
def run_on_cluster(cluster_config: Union[dict, str], *, cmd: Optional[str]=None, run_env: str='auto', tmux: bool=False, stop: bool=False, no_config_cache: bool=False, port_forward: Optional[commands.Port_forward]=None, with_output: bool=False) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    'Runs a command on the specified cluster.\n\n    Args:\n        cluster_config (Union[str, dict]): Either the config dict of the\n            cluster, or a path pointing to a file containing the config.\n        cmd: the command to run, or None for a no-op command.\n        run_env: whether to run the command on the host or in a\n            container. Select between "auto", "host" and "docker".\n        tmux: whether to run in a tmux session\n        stop: whether to stop the cluster after command run\n        no_config_cache: Whether to disable the config cache and fully\n            resolve all environment settings from the Cloud provider again.\n        port_forward ( (int,int) or list[(int,int)]): port(s) to forward.\n        with_output: Whether to capture command output.\n\n    Returns:\n        The output of the command as a string.\n    '
    with _as_config_file(cluster_config) as config_file:
        return commands.exec_cluster(config_file, cmd=cmd, run_env=run_env, screen=False, tmux=tmux, stop=stop, start=False, override_cluster_name=None, no_config_cache=no_config_cache, port_forward=port_forward, with_output=with_output)

@DeveloperAPI
def rsync(cluster_config: Union[dict, str], *, source: Optional[str], target: Optional[str], down: bool, ip_address: str=None, use_internal_ip: bool=False, no_config_cache: bool=False, should_bootstrap: bool=True):
    if False:
        print('Hello World!')
    "Rsyncs files to or from the cluster.\n\n    Args:\n        cluster_config (Union[str, dict]): Either the config dict of the\n            cluster, or a path pointing to a file containing the config.\n        source: rsync source argument.\n        target: rsync target argument.\n        down: whether we're syncing remote -> local.\n        ip_address: Address of node.\n        use_internal_ip: Whether the provided ip_address is\n            public or private.\n        no_config_cache: Whether to disable the config cache and fully\n            resolve all environment settings from the Cloud provider again.\n        should_bootstrap: whether to bootstrap cluster config before syncing\n\n    Raises:\n        RuntimeError if the cluster head node is not found.\n    "
    with _as_config_file(cluster_config) as config_file:
        return commands.rsync(config_file=config_file, source=source, target=target, override_cluster_name=None, down=down, ip_address=ip_address, use_internal_ip=use_internal_ip, no_config_cache=no_config_cache, all_nodes=False, should_bootstrap=should_bootstrap)

@DeveloperAPI
def get_head_node_ip(cluster_config: Union[dict, str]) -> str:
    if False:
        print('Hello World!')
    'Returns head node IP for given configuration file if exists.\n\n    Args:\n        cluster_config (Union[str, dict]): Either the config dict of the\n            cluster, or a path pointing to a file containing the config.\n\n    Returns:\n        The ip address of the cluster head node.\n\n    Raises:\n        RuntimeError if the cluster is not found.\n    '
    with _as_config_file(cluster_config) as config_file:
        return commands.get_head_node_ip(config_file)

@DeveloperAPI
def get_worker_node_ips(cluster_config: Union[dict, str]) -> List[str]:
    if False:
        while True:
            i = 10
    'Returns worker node IPs for given configuration file.\n\n    Args:\n        cluster_config (Union[str, dict]): Either the config dict of the\n            cluster, or a path pointing to a file containing the config.\n\n    Returns:\n        List of worker node ip addresses.\n\n    Raises:\n        RuntimeError if the cluster is not found.\n    '
    with _as_config_file(cluster_config) as config_file:
        return commands.get_worker_node_ips(config_file)

@DeveloperAPI
def request_resources(num_cpus: Optional[int]=None, bundles: Optional[List[dict]]=None) -> None:
    if False:
        return 10
    'Command the autoscaler to scale to accommodate the specified requests.\n\n    The cluster will immediately attempt to scale to accommodate the requested\n    resources, bypassing normal upscaling speed constraints. This takes into\n    account existing resource usage.\n\n    For example, suppose you call ``request_resources(num_cpus=100)`` and\n    there are 45 currently running tasks, each requiring 1 CPU. Then, enough\n    nodes will be added so up to 100 tasks can run concurrently. It does\n    **not** add enough nodes so that 145 tasks can run.\n\n    This call is only a hint to the autoscaler. The actual resulting cluster\n    size may be slightly larger or smaller than expected depending on the\n    internal bin packing algorithm and max worker count restrictions.\n\n    Args:\n        num_cpus: Scale the cluster to ensure this number of CPUs are\n            available. This request is persistent until another call to\n            request_resources() is made to override.\n        bundles (List[ResourceDict]): Scale the cluster to ensure this set of\n            resource shapes can fit. This request is persistent until another\n            call to request_resources() is made to override.\n\n    Examples:\n        >>> from ray.autoscaler.sdk import request_resources\n        >>> # Request 1000 CPUs.\n        >>> request_resources(num_cpus=1000) # doctest: +SKIP\n        >>> # Request 64 CPUs and also fit a 1-GPU/4-CPU task.\n        >>> request_resources( # doctest: +SKIP\n        ...     num_cpus=64, bundles=[{"GPU": 1, "CPU": 4}])\n        >>> # Same as requesting num_cpus=3.\n        >>> request_resources( # doctest: +SKIP\n        ...     bundles=[{"CPU": 1}, {"CPU": 1}, {"CPU": 1}])\n    '
    if num_cpus is not None and (not isinstance(num_cpus, int)):
        raise TypeError('num_cpus should be of type int.')
    if bundles is not None:
        if isinstance(bundles, List):
            for bundle in bundles:
                if isinstance(bundle, Dict):
                    for key in bundle.keys():
                        if not (isinstance(key, str) and isinstance(bundle[key], int)):
                            raise TypeError('each bundle key should be str and value as int.')
                else:
                    raise TypeError('each bundle should be a Dict.')
        else:
            raise TypeError('bundles should be of type List')
    return commands.request_resources(num_cpus, bundles)

@DeveloperAPI
def configure_logging(log_style: Optional[str]=None, color_mode: Optional[str]=None, verbosity: Optional[int]=None):
    if False:
        return 10
    'Configures logging for cluster command calls.\n\n    Args:\n        log_style: If \'pretty\', outputs with formatting and color.\n            If \'record\', outputs record-style without formatting.\n            \'auto\' defaults to \'pretty\', and disables pretty logging\n            if stdin is *not* a TTY. Defaults to "auto".\n        color_mode (str):\n            Can be "true", "false", or "auto".\n\n            Enables or disables `colorful`.\n\n            If `color_mode` is "auto", is set to `not stdout.isatty()`\n        vebosity (int):\n            Output verbosity (0, 1, 2, 3).\n\n            Low verbosity will disable `verbose` and `very_verbose` messages.\n\n    '
    cli_logger.configure(log_style=log_style, color_mode=color_mode, verbosity=verbosity)

@contextmanager
@DeveloperAPI
def _as_config_file(cluster_config: Union[dict, str]) -> Iterator[str]:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(cluster_config, dict):
        tmp = tempfile.NamedTemporaryFile('w', prefix='autoscaler-sdk-tmp-')
        tmp.write(json.dumps(cluster_config))
        tmp.flush()
        cluster_config = tmp.name
    if not os.path.exists(cluster_config):
        raise ValueError('Cluster config not found {}'.format(cluster_config))
    yield cluster_config

@DeveloperAPI
def bootstrap_config(cluster_config: Dict[str, Any], no_config_cache: bool=False) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    'Validate and add provider-specific fields to the config. For example,\n    IAM/authentication may be added here.'
    return commands._bootstrap_config(cluster_config, no_config_cache)

@DeveloperAPI
def fillout_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    'Fillout default values for a cluster_config based on the provider.'
    from ray.autoscaler._private.util import fillout_defaults
    return fillout_defaults(config)

@DeveloperAPI
def register_callback_handler(event_name: str, callback: Union[Callable[[Dict], None], List[Callable[[Dict], None]]]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Registers a callback handler for autoscaler events.\n\n    Args:\n        event_name: Event that callback should be called on. See\n            CreateClusterEvent for details on the events available to be\n            registered against.\n        callback: Callable object that is invoked\n            when specified event occurs.\n    '
    global_event_system.add_callback_handler(event_name, callback)

@DeveloperAPI
def get_docker_host_mount_location(cluster_name: str) -> str:
    if False:
        return 10
    'Return host path that Docker mounts attach to.'
    docker_mount_prefix = '/tmp/ray_tmp_mount/{cluster_name}'
    return docker_mount_prefix.format(cluster_name=cluster_name)