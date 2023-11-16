import os
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import List, Optional, Sequence, Tuple
import yaml
import ray
from ray.autoscaler._private.cli_logger import cli_logger
from ray.autoscaler._private.providers import _get_node_provider
from ray.autoscaler.tags import NODE_KIND_HEAD, NODE_KIND_WORKER, TAG_RAY_NODE_KIND
import psutil
MAX_PARALLEL_SSH_WORKERS = 8
DEFAULT_SSH_USER = 'ubuntu'
DEFAULT_SSH_KEYS = ['~/ray_bootstrap_key.pem', '~/.ssh/ray-autoscaler_2_us-west-2.pem']

class CommandFailed(RuntimeError):
    pass

class LocalCommandFailed(CommandFailed):
    pass

class RemoteCommandFailed(CommandFailed):
    pass

class GetParameters:

    def __init__(self, logs: bool=True, debug_state: bool=True, pip: bool=True, processes: bool=True, processes_verbose: bool=True, processes_list: Optional[List[Tuple[str, bool]]]=None):
        if False:
            for i in range(10):
                print('nop')
        self.logs = logs
        self.debug_state = debug_state
        self.pip = pip
        self.processes = processes
        self.processes_verbose = processes_verbose
        self.processes_list = processes_list

class Node:
    """Node (as in "machine")"""

    def __init__(self, host: str, ssh_user: str='ubuntu', ssh_key: str='~/ray_bootstrap_key.pem', docker_container: Optional[str]=None, is_head: bool=False):
        if False:
            while True:
                i = 10
        self.host = host
        self.ssh_user = ssh_user
        self.ssh_key = ssh_key
        self.docker_container = docker_container
        self.is_head = is_head

class Archive:
    """Archive object to collect and compress files into a single file.

    Objects of this class can be passed around to different data collection
    functions. These functions can use the :meth:`subdir` method to add
    files to a sub directory of the archive.

    """

    def __init__(self, file: Optional[str]=None):
        if False:
            return 10
        self.file = file or tempfile.mkstemp(prefix='ray_logs_', suffix='.tar.gz')[1]
        self.tar = None
        self._lock = threading.Lock()

    @property
    def is_open(self):
        if False:
            print('Hello World!')
        return bool(self.tar)

    def open(self):
        if False:
            while True:
                i = 10
        self.tar = tarfile.open(self.file, 'w:gz')

    def close(self):
        if False:
            print('Hello World!')
        self.tar.close()
        self.tar = None

    def __enter__(self):
        if False:
            return 10
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        self.close()

    @contextmanager
    def subdir(self, subdir: str, root: Optional[str]='/'):
        if False:
            i = 10
            return i + 15
        'Open a context to add files to the archive.\n\n        Example:\n\n            .. code-block:: python\n\n                with Archive("file.tar.gz") as archive:\n                    with archive.subdir("logfiles", root="/tmp/logs") as sd:\n                        # Will be added as `logfiles/nested/file.txt`\n                        sd.add("/tmp/logs/nested/file.txt")\n\n        Args:\n            subdir: Subdir to which to add files to. Calling the\n                ``add(path)`` command will place files into the ``subdir``\n                directory of the archive.\n            root: Root path. Files without an explicit ``arcname``\n                will be named relatively to this path.\n\n        Yields:\n            A context object that can be used to add files to the archive.\n        '
        root = os.path.abspath(root)

        class _Context:

            @staticmethod
            def add(path: str, arcname: Optional[str]=None):
                if False:
                    while True:
                        i = 10
                path = os.path.abspath(path)
                arcname = arcname or os.path.join(subdir, os.path.relpath(path, root))
                self._lock.acquire()
                self.tar.add(path, arcname=arcname)
                self._lock.release()
        yield _Context()

def get_local_ray_logs(archive: Archive, exclude: Optional[Sequence[str]]=None, session_log_dir: str='/tmp/ray/session_latest') -> Archive:
    if False:
        while True:
            i = 10
    'Copy local log files into an archive.\n\n    Args:\n        archive: Archive object to add log files to.\n        exclude (Sequence[str]): Sequence of regex patterns. Files that match\n            any of these patterns will not be included in the archive.\n        session_dir: Path to the Ray session files. Defaults to\n            ``/tmp/ray/session_latest``\n\n    Returns:\n        Open archive object.\n\n    '
    if not archive.is_open:
        archive.open()
    exclude = exclude or []
    session_log_dir = os.path.join(os.path.expanduser(session_log_dir), 'logs')
    with archive.subdir('logs', root=session_log_dir) as sd:
        for (root, dirs, files) in os.walk(session_log_dir):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, start=session_log_dir)
                if any((re.match(pattern, rel_path) for pattern in exclude)):
                    continue
                sd.add(file_path)
    return archive

def get_local_debug_state(archive: Archive, session_dir: str='/tmp/ray/session_latest') -> Archive:
    if False:
        for i in range(10):
            print('nop')
    'Copy local log files into an archive.\n\n    Args:\n        archive: Archive object to add log files to.\n        session_dir: Path to the Ray session files. Defaults to\n            ``/tmp/ray/session_latest``\n\n    Returns:\n        Open archive object.\n\n    '
    if not archive.is_open:
        archive.open()
    session_dir = os.path.expanduser(session_dir)
    debug_state_file = os.path.join(session_dir, 'logs/debug_state.txt')
    if not os.path.exists(debug_state_file):
        raise LocalCommandFailed('No `debug_state.txt` file found.')
    with archive.subdir('', root=session_dir) as sd:
        sd.add(debug_state_file)
    return archive

def get_local_pip_packages(archive: Archive):
    if False:
        return 10
    'Get currently installed pip packages and write into an archive.\n\n    Args:\n        archive: Archive object to add meta files to.\n\n    Returns:\n        Open archive object.\n    '
    if not archive.is_open:
        archive.open()
    try:
        from pip._internal.operations import freeze
    except ImportError:
        from pip.operations import freeze
    with tempfile.NamedTemporaryFile('wt') as fp:
        for line in freeze.freeze():
            fp.writelines([line, '\n'])
        fp.flush()
        with archive.subdir('') as sd:
            sd.add(fp.name, 'pip_packages.txt')
    return archive

def get_local_ray_processes(archive: Archive, processes: Optional[List[Tuple[str, bool]]]=None, verbose: bool=False):
    if False:
        i = 10
        return i + 15
    'Get the status of all the relevant ray processes.\n    Args:\n        archive: Archive object to add process info files to.\n        processes: List of processes to get information on. The first\n            element of the tuple is a string to filter by, and the second\n            element is a boolean indicating if we should filter by command\n            name (True) or command line including parameters (False)\n        verbose: If True, show entire executable command line.\n            If False, show just the first term.\n    Returns:\n        Open archive object.\n    '
    if not processes:
        from ray.autoscaler._private.constants import RAY_PROCESSES
        processes = RAY_PROCESSES
    process_infos = []
    for process in psutil.process_iter(['pid', 'name', 'cmdline', 'status']):
        try:
            with process.oneshot():
                cmdline = ' '.join(process.cmdline())
                process_infos.append(({'executable': cmdline if verbose else cmdline.split('--', 1)[0][:-1], 'name': process.name(), 'pid': process.pid, 'status': process.status()}, process.cmdline()))
        except Exception as exc:
            raise LocalCommandFailed(exc) from exc
    relevant_processes = {}
    for (process_dict, cmdline) in process_infos:
        for (keyword, filter_by_cmd) in processes:
            if filter_by_cmd:
                corpus = process_dict['name']
            else:
                corpus = subprocess.list2cmdline(cmdline)
            if keyword in corpus and process_dict['pid'] not in relevant_processes:
                relevant_processes[process_dict['pid']] = process_dict
    with tempfile.NamedTemporaryFile('wt') as fp:
        for line in relevant_processes.values():
            fp.writelines([yaml.dump(line), '\n'])
        fp.flush()
        with archive.subdir('meta') as sd:
            sd.add(fp.name, 'process_info.txt')
    return archive

def get_all_local_data(archive: Archive, parameters: GetParameters):
    if False:
        i = 10
        return i + 15
    'Get all local data.\n\n    Gets:\n        - The Ray logs of the latest session\n        - The currently installed pip packages\n\n    Args:\n        archive: Archive object to add meta files to.\n        parameters: Parameters (settings) for getting data.\n\n    Returns:\n        Open archive object.\n    '
    if not archive.is_open:
        archive.open()
    if parameters.logs:
        try:
            get_local_ray_logs(archive=archive)
        except LocalCommandFailed as exc:
            cli_logger.error(exc)
    if parameters.debug_state:
        try:
            get_local_debug_state(archive=archive)
        except LocalCommandFailed as exc:
            cli_logger.error(exc)
    if parameters.pip:
        try:
            get_local_pip_packages(archive=archive)
        except LocalCommandFailed as exc:
            cli_logger.error(exc)
    if parameters.processes:
        try:
            get_local_ray_processes(archive=archive, processes=parameters.processes_list, verbose=parameters.processes_verbose)
        except LocalCommandFailed as exc:
            cli_logger.error(exc)
    return archive

def _wrap(items: List[str], quotes="'"):
    if False:
        for i in range(10):
            print('nop')
    return f"{quotes}{' '.join(items)}{quotes}"

def create_and_get_archive_from_remote_node(remote_node: Node, parameters: GetParameters, script_path: str='ray') -> Optional[str]:
    if False:
        i = 10
        return i + 15
    "Create an archive containing logs on a remote node and transfer.\n\n    This will call ``ray local-dump --stream`` on the remote\n    node. The resulting file will be saved locally in a temporary file and\n    returned.\n\n    Args:\n        remote_node: Remote node to gather archive from.\n        script_path: Path to this script on the remote node.\n        parameters: Parameters (settings) for getting data.\n\n    Returns:\n        Path to a temporary file containing the node's collected data.\n\n    "
    cmd = ['ssh', '-o StrictHostKeyChecking=no', '-o UserKnownHostsFile=/dev/null', '-o LogLevel=ERROR', '-i', remote_node.ssh_key, f'{remote_node.ssh_user}@{remote_node.host}']
    if remote_node.docker_container:
        cmd += ['docker', 'exec', remote_node.docker_container]
    collect_cmd = [script_path, 'local-dump', '--stream']
    collect_cmd += ['--logs'] if parameters.logs else ['--no-logs']
    collect_cmd += ['--debug-state'] if parameters.debug_state else ['--no-debug-state']
    collect_cmd += ['--pip'] if parameters.pip else ['--no-pip']
    collect_cmd += ['--processes'] if parameters.processes else ['--no-processes']
    if parameters.processes:
        collect_cmd += ['--processes-verbose'] if parameters.processes_verbose else ['--no-proccesses-verbose']
    cmd += ['/bin/bash', '-c', _wrap(collect_cmd, quotes='"')]
    cat = 'node' if not remote_node.is_head else 'head'
    cli_logger.print(f'Collecting data from remote node: {remote_node.host}')
    tmp = tempfile.mkstemp(prefix=f'ray_{cat}_{remote_node.host}_', suffix='.tar.gz')[1]
    with open(tmp, 'wb') as fp:
        try:
            subprocess.check_call(cmd, stdout=fp, stderr=sys.stderr)
        except subprocess.CalledProcessError as exc:
            raise RemoteCommandFailed(f"Gathering logs from remote node failed: {' '.join(cmd)}") from exc
    return tmp

def create_and_add_remote_data_to_local_archive(archive: Archive, remote_node: Node, parameters: GetParameters):
    if False:
        return 10
    'Create and get data from remote node and add to local archive.\n\n    Args:\n        archive: Archive object to add remote data to.\n        remote_node: Remote node to gather archive from.\n        parameters: Parameters (settings) for getting data.\n\n    Returns:\n        Open archive object.\n    '
    tmp = create_and_get_archive_from_remote_node(remote_node, parameters)
    if not archive.is_open:
        archive.open()
    cat = 'node' if not remote_node.is_head else 'head'
    with archive.subdir('', root=os.path.dirname(tmp)) as sd:
        sd.add(tmp, arcname=f'ray_{cat}_{remote_node.host}.tar.gz')
    return archive

def create_and_add_local_data_to_local_archive(archive: Archive, parameters: GetParameters):
    if False:
        return 10
    'Create and get data from this node and add to archive.\n\n    Args:\n        archive: Archive object to add remote data to.\n        parameters: Parameters (settings) for getting data.\n\n    Returns:\n        Open archive object.\n    '
    with Archive() as local_data_archive:
        get_all_local_data(local_data_archive, parameters)
    if not archive.is_open:
        archive.open()
    with archive.subdir('', root=os.path.dirname(local_data_archive.file)) as sd:
        sd.add(local_data_archive.file, arcname='local_node.tar.gz')
    os.remove(local_data_archive.file)
    return archive

def create_archive_for_remote_nodes(archive: Archive, remote_nodes: Sequence[Node], parameters: GetParameters):
    if False:
        for i in range(10):
            print('nop')
    'Create an archive combining data from the remote nodes.\n\n    This will parallelize calls to get data from remote nodes.\n\n    Args:\n        archive: Archive object to add remote data to.\n        remote_nodes (Sequence[Node]): Sequence of remote nodes.\n        parameters: Parameters (settings) for getting data.\n\n    Returns:\n        Open archive object.\n\n    '
    if not archive.is_open:
        archive.open()
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SSH_WORKERS) as executor:
        for remote_node in remote_nodes:
            executor.submit(create_and_add_remote_data_to_local_archive, archive=archive, remote_node=remote_node, parameters=parameters)
    return archive

def create_archive_for_local_and_remote_nodes(archive: Archive, remote_nodes: Sequence[Node], parameters: GetParameters):
    if False:
        i = 10
        return i + 15
    'Create an archive combining data from the local and remote nodes.\n\n    This will parallelize calls to get data from remote nodes.\n\n    Args:\n        archive: Archive object to add data to.\n        remote_nodes (Sequence[Node]): Sequence of remote nodes.\n        parameters: Parameters (settings) for getting data.\n\n    Returns:\n        Open archive object.\n\n    '
    if not archive.is_open:
        archive.open()
    try:
        create_and_add_local_data_to_local_archive(archive, parameters)
    except CommandFailed as exc:
        cli_logger.error(exc)
    create_archive_for_remote_nodes(archive, remote_nodes, parameters)
    cli_logger.print(f'Collected data from local node and {len(remote_nodes)} remote nodes.')
    return archive

def get_info_from_ray_cluster_config(cluster_config: str) -> Tuple[List[str], str, str, Optional[str], Optional[str]]:
    if False:
        while True:
            i = 10
    'Get information from Ray cluster config.\n\n    Return list of host IPs, ssh user, ssh key file, and optional docker\n    container.\n\n    Args:\n        cluster_config: Path to ray cluster config.\n\n    Returns:\n        Tuple of list of host IPs, ssh user name, ssh key file path,\n            optional docker container name, optional cluster name.\n    '
    from ray.autoscaler._private.commands import _bootstrap_config
    cli_logger.print(f'Retrieving cluster information from ray cluster file: {cluster_config}')
    cluster_config = os.path.expanduser(cluster_config)
    config = yaml.safe_load(open(cluster_config).read())
    config = _bootstrap_config(config, no_config_cache=True)
    provider = _get_node_provider(config['provider'], config['cluster_name'])
    head_nodes = provider.non_terminated_nodes({TAG_RAY_NODE_KIND: NODE_KIND_HEAD})
    worker_nodes = provider.non_terminated_nodes({TAG_RAY_NODE_KIND: NODE_KIND_WORKER})
    hosts = [provider.external_ip(node) for node in head_nodes + worker_nodes]
    ssh_user = config['auth']['ssh_user']
    ssh_key = config['auth']['ssh_private_key']
    docker = None
    docker_config = config.get('docker', None)
    if docker_config:
        docker = docker_config.get('container_name', None)
    cluster_name = config.get('cluster_name', None)
    return (hosts, ssh_user, ssh_key, docker, cluster_name)

def _info_from_params(cluster: Optional[str]=None, host: Optional[str]=None, ssh_user: Optional[str]=None, ssh_key: Optional[str]=None, docker: Optional[str]=None):
    if False:
        while True:
            i = 10
    'Parse command line arguments.\n\n    Note: This returns a list of hosts, not a comma separated string!\n    '
    if not host and (not cluster):
        bootstrap_config = os.path.expanduser('~/ray_bootstrap_config.yaml')
        if os.path.exists(bootstrap_config):
            cluster = bootstrap_config
            cli_logger.warning(f'Detected cluster config file at {cluster}. If this is incorrect, specify with `ray cluster-dump <config>`')
    elif cluster:
        cluster = os.path.expanduser(cluster)
    cluster_name = None
    if cluster:
        (h, u, k, d, cluster_name) = get_info_from_ray_cluster_config(cluster)
        ssh_user = ssh_user or u
        ssh_key = ssh_key or k
        docker = docker or d
        hosts = host.split(',') if host else h
        if not hosts:
            raise LocalCommandFailed(f'Invalid cluster file or cluster has no running nodes: {cluster}')
    elif host:
        hosts = host.split(',')
    else:
        raise LocalCommandFailed('You need to either specify a `<cluster_config>` or `--host`.')
    if not ssh_user:
        ssh_user = DEFAULT_SSH_USER
        cli_logger.warning(f'Using default SSH user `{ssh_user}`. If this is incorrect, specify with `--ssh-user <user>`')
    if not ssh_key:
        for cand_key in DEFAULT_SSH_KEYS:
            cand_key_file = os.path.expanduser(cand_key)
            if os.path.exists(cand_key_file):
                ssh_key = cand_key_file
                cli_logger.warning(f'Auto detected SSH key file: {ssh_key}. If this is incorrect, specify with `--ssh-key <key>`')
                break
    return (cluster, hosts, ssh_user, ssh_key, docker, cluster_name)