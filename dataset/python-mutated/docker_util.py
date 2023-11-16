"""Functions for accessing docker via the docker cli."""
from __future__ import annotations
import dataclasses
import enum
import json
import os
import pathlib
import re
import socket
import time
import urllib.parse
import typing as t
from .util import ApplicationError, common_environment, display, find_executable, SubprocessError, cache, OutputStream
from .util_common import run_command, raw_command
from .config import CommonConfig
from .thread import mutex, named_lock
from .cgroup import CGroupEntry, MountEntry, MountType
DOCKER_COMMANDS = ['docker', 'podman']
UTILITY_IMAGE = 'quay.io/ansible/ansible-test-utility-container:2.0.0'
MAX_NUM_OPEN_FILES = 10240
LOGINUID_NOT_SET = 4294967295

class DockerInfo:
    """The results of `docker info` and `docker version` for the container runtime."""

    @classmethod
    def init(cls, args: CommonConfig) -> DockerInfo:
        if False:
            while True:
                i = 10
        'Initialize and return a DockerInfo instance.'
        command = require_docker().command
        info_stdout = docker_command(args, ['info', '--format', '{{ json . }}'], capture=True, always=True)[0]
        info = json.loads(info_stdout)
        if (server_errors := info.get('ServerErrors')):
            raise ApplicationError('Unable to get container host information: ' + '\n'.join(server_errors))
        version_stdout = docker_command(args, ['version', '--format', '{{ json . }}'], capture=True, always=True)[0]
        version = json.loads(version_stdout)
        info = DockerInfo(args, command, info, version)
        return info

    def __init__(self, args: CommonConfig, engine: str, info: dict[str, t.Any], version: dict[str, t.Any]) -> None:
        if False:
            while True:
                i = 10
        self.args = args
        self.engine = engine
        self.info = info
        self.version = version

    @property
    def client(self) -> dict[str, t.Any]:
        if False:
            i = 10
            return i + 15
        'The client version details.'
        client = self.version.get('Client')
        if not client:
            raise ApplicationError('Unable to get container host client information.')
        return client

    @property
    def server(self) -> dict[str, t.Any]:
        if False:
            return 10
        'The server version details.'
        server = self.version.get('Server')
        if not server:
            if self.engine == 'podman':
                return self.client
            raise ApplicationError('Unable to get container host server information.')
        return server

    @property
    def client_version(self) -> str:
        if False:
            print('Hello World!')
        'The client version.'
        return self.client['Version']

    @property
    def server_version(self) -> str:
        if False:
            print('Hello World!')
        'The server version.'
        return self.server['Version']

    @property
    def client_major_minor_version(self) -> tuple[int, int]:
        if False:
            i = 10
            return i + 15
        'The client major and minor version.'
        (major, minor) = self.client_version.split('.')[:2]
        return (int(major), int(minor))

    @property
    def server_major_minor_version(self) -> tuple[int, int]:
        if False:
            return 10
        'The server major and minor version.'
        (major, minor) = self.server_version.split('.')[:2]
        return (int(major), int(minor))

    @property
    def cgroupns_option_supported(self) -> bool:
        if False:
            print('Hello World!')
        'Return True if the `--cgroupns` option is supported, otherwise return False.'
        if self.engine == 'docker':
            return self.client_major_minor_version >= (20, 10) and self.server_major_minor_version >= (20, 10)
        raise NotImplementedError(self.engine)

    @property
    def cgroup_version(self) -> int:
        if False:
            print('Hello World!')
        'The cgroup version of the container host.'
        info = self.info
        host = info.get('host')
        if host:
            return int(host['cgroupVersion'].lstrip('v'))
        try:
            return int(info['CgroupVersion'])
        except KeyError:
            pass
        if self.server_major_minor_version < (20, 10):
            return 1
        message = f'The Docker client version is {self.client_version}. The Docker server version is {self.server_version}. Upgrade your Docker client to version 20.10 or later.'
        if detect_host_properties(self.args).cgroup_v2:
            raise ApplicationError(f'Unsupported Docker client and server combination using cgroup v2. {message}')
        display.warning(f'Detected Docker server cgroup v1 using probing. {message}', unique=True)
        return 1

    @property
    def docker_desktop_wsl2(self) -> bool:
        if False:
            i = 10
            return i + 15
        'Return True if Docker Desktop integrated with WSL2 is detected, otherwise False.'
        info = self.info
        kernel_version = info.get('KernelVersion')
        operating_system = info.get('OperatingSystem')
        dd_wsl2 = kernel_version and kernel_version.endswith('-WSL2') and (operating_system == 'Docker Desktop')
        return dd_wsl2

    @property
    def description(self) -> str:
        if False:
            print('Hello World!')
        'Describe the container runtime.'
        tags = dict(client=self.client_version, server=self.server_version, cgroup=f'v{self.cgroup_version}')
        labels = [self.engine] + [f'{key}={value}' for (key, value) in tags.items()]
        if self.docker_desktop_wsl2:
            labels.append('DD+WSL2')
        return f"Container runtime: {' '.join(labels)}"

@mutex
def get_docker_info(args: CommonConfig) -> DockerInfo:
    if False:
        i = 10
        return i + 15
    'Return info for the current container runtime. The results are cached.'
    try:
        return get_docker_info.info
    except AttributeError:
        pass
    info = DockerInfo.init(args)
    display.info(info.description, verbosity=1)
    get_docker_info.info = info
    return info

class SystemdControlGroupV1Status(enum.Enum):
    """The state of the cgroup v1 systemd hierarchy on the container host."""
    SUBSYSTEM_MISSING = 'The systemd cgroup subsystem was not found.'
    FILESYSTEM_NOT_MOUNTED = 'The "/sys/fs/cgroup/systemd" filesystem is not mounted.'
    MOUNT_TYPE_NOT_CORRECT = 'The "/sys/fs/cgroup/systemd" mount type is not correct.'
    VALID = 'The "/sys/fs/cgroup/systemd" mount is valid.'

@dataclasses.dataclass(frozen=True)
class ContainerHostProperties:
    """Container host properties detected at run time."""
    audit_code: str
    max_open_files: int
    loginuid: t.Optional[int]
    cgroup_v1: SystemdControlGroupV1Status
    cgroup_v2: bool

@mutex
def detect_host_properties(args: CommonConfig) -> ContainerHostProperties:
    if False:
        for i in range(10):
            print('nop')
    "\n    Detect and return properties of the container host.\n\n    The information collected is:\n\n      - The errno result from attempting to query the container host's audit status.\n      - The max number of open files supported by the container host to run containers.\n        This value may be capped to the maximum value used by ansible-test.\n        If the value is below the desired limit, a warning is displayed.\n      - The loginuid used by the container host to run containers, or None if the audit subsystem is unavailable.\n      - The cgroup subsystems registered with the Linux kernel.\n      - The mounts visible within a container.\n      - The status of the systemd cgroup v1 hierarchy.\n\n    This information is collected together to reduce the number of container runs to probe the container host.\n    "
    try:
        return detect_host_properties.properties
    except AttributeError:
        pass
    single_line_commands = ('audit-status', 'cat /proc/sys/fs/nr_open', 'ulimit -Hn', '(cat /proc/1/loginuid; echo)')
    multi_line_commands = (' && '.join(single_line_commands), 'cat /proc/1/cgroup', 'cat /proc/1/mountinfo')
    options = ['--volume', '/sys/fs/cgroup:/probe:ro']
    cmd = ['sh', '-c', ' && echo "-" && '.join(multi_line_commands)]
    stdout = run_utility_container(args, 'ansible-test-probe', cmd, options)[0]
    if args.explain:
        return ContainerHostProperties(audit_code='???', max_open_files=MAX_NUM_OPEN_FILES, loginuid=LOGINUID_NOT_SET, cgroup_v1=SystemdControlGroupV1Status.VALID, cgroup_v2=False)
    blocks = stdout.split('\n-\n')
    values = blocks[0].split('\n')
    audit_parts = values[0].split(' ', 1)
    audit_status = int(audit_parts[0])
    audit_code = audit_parts[1]
    system_limit = int(values[1])
    hard_limit = int(values[2])
    loginuid = int(values[3]) if values[3] else None
    cgroups = CGroupEntry.loads(blocks[1])
    mounts = MountEntry.loads(blocks[2])
    if hard_limit < MAX_NUM_OPEN_FILES and hard_limit < system_limit and (require_docker().command == 'docker'):
        options = ['--ulimit', f'nofile={min(system_limit, MAX_NUM_OPEN_FILES)}']
        cmd = ['sh', '-c', 'ulimit -Hn']
        try:
            stdout = run_utility_container(args, 'ansible-test-ulimit', cmd, options)[0]
        except SubprocessError as ex:
            display.warning(str(ex))
        else:
            hard_limit = int(stdout)
    subsystems = set((cgroup.subsystem for cgroup in cgroups))
    mount_types = {mount.path: mount.type for mount in mounts}
    if 'systemd' not in subsystems:
        cgroup_v1 = SystemdControlGroupV1Status.SUBSYSTEM_MISSING
    elif not (mount_type := mount_types.get(pathlib.PurePosixPath('/probe/systemd'))):
        cgroup_v1 = SystemdControlGroupV1Status.FILESYSTEM_NOT_MOUNTED
    elif mount_type != MountType.CGROUP_V1:
        cgroup_v1 = SystemdControlGroupV1Status.MOUNT_TYPE_NOT_CORRECT
    else:
        cgroup_v1 = SystemdControlGroupV1Status.VALID
    cgroup_v2 = mount_types.get(pathlib.PurePosixPath('/probe')) == MountType.CGROUP_V2
    display.info(f'Container host audit status: {audit_code} ({audit_status})', verbosity=1)
    display.info(f'Container host max open files: {hard_limit}', verbosity=1)
    display.info(f"Container loginuid: {(loginuid if loginuid is not None else 'unavailable')}{(' (not set)' if loginuid == LOGINUID_NOT_SET else '')}", verbosity=1)
    if hard_limit < MAX_NUM_OPEN_FILES:
        display.warning(f'Unable to set container max open files to {MAX_NUM_OPEN_FILES}. Using container host limit of {hard_limit} instead.')
    else:
        hard_limit = MAX_NUM_OPEN_FILES
    properties = ContainerHostProperties(audit_code=audit_code, max_open_files=hard_limit, loginuid=loginuid, cgroup_v1=cgroup_v1, cgroup_v2=cgroup_v2)
    detect_host_properties.properties = properties
    return properties

def get_session_container_name(args: CommonConfig, name: str) -> str:
    if False:
        return 10
    'Return the given container name with the current test session name applied to it.'
    return f'{name}-{args.session_name}'

def run_utility_container(args: CommonConfig, name: str, cmd: list[str], options: list[str], data: t.Optional[str]=None) -> tuple[t.Optional[str], t.Optional[str]]:
    if False:
        print('Hello World!')
    'Run the specified command using the ansible-test utility container, returning stdout and stderr.'
    name = get_session_container_name(args, name)
    options = options + ['--name', name, '--rm']
    if data:
        options.append('-i')
    docker_pull(args, UTILITY_IMAGE)
    return docker_run(args, UTILITY_IMAGE, options, cmd, data)

class DockerCommand:
    """Details about the available docker command."""

    def __init__(self, command: str, executable: str, version: str) -> None:
        if False:
            while True:
                i = 10
        self.command = command
        self.executable = executable
        self.version = version

    @staticmethod
    def detect() -> t.Optional[DockerCommand]:
        if False:
            return 10
        'Detect and return the available docker command, or None.'
        if os.environ.get('ANSIBLE_TEST_PREFER_PODMAN'):
            commands = list(reversed(DOCKER_COMMANDS))
        else:
            commands = DOCKER_COMMANDS
        for command in commands:
            executable = find_executable(command, required=False)
            if executable:
                version = raw_command([command, '-v'], env=docker_environment(), capture=True)[0].strip()
                if command == 'docker' and 'podman' in version:
                    continue
                display.info('Detected "%s" container runtime version: %s' % (command, version), verbosity=1)
                return DockerCommand(command, executable, version)
        return None

def require_docker() -> DockerCommand:
    if False:
        for i in range(10):
            print('nop')
    'Return the docker command to invoke. Raises an exception if docker is not available.'
    if (command := get_docker_command()):
        return command
    raise ApplicationError(f"No container runtime detected. Supported commands: {', '.join(DOCKER_COMMANDS)}")

@cache
def get_docker_command() -> t.Optional[DockerCommand]:
    if False:
        return 10
    'Return the docker command to invoke, or None if docker is not available.'
    return DockerCommand.detect()

def docker_available() -> bool:
    if False:
        i = 10
        return i + 15
    'Return True if docker is available, otherwise return False.'
    return bool(get_docker_command())

@cache
def get_docker_host_ip() -> str:
    if False:
        while True:
            i = 10
    'Return the IP of the Docker host.'
    docker_host_ip = socket.gethostbyname(get_docker_hostname())
    display.info('Detected docker host IP: %s' % docker_host_ip, verbosity=1)
    return docker_host_ip

@cache
def get_docker_hostname() -> str:
    if False:
        for i in range(10):
            print('nop')
    'Return the hostname of the Docker service.'
    docker_host = os.environ.get('DOCKER_HOST')
    if docker_host and docker_host.startswith(('tcp://', 'ssh://')):
        try:
            hostname = urllib.parse.urlparse(docker_host)[1].split(':')[0]
            display.info('Detected Docker host: %s' % hostname, verbosity=1)
        except ValueError:
            hostname = 'localhost'
            display.warning('Could not parse DOCKER_HOST environment variable "%s", falling back to localhost.' % docker_host)
    else:
        hostname = 'localhost'
        display.info('Assuming Docker is available on localhost.', verbosity=1)
    return hostname

@cache
def get_podman_host_ip() -> str:
    if False:
        return 10
    'Return the IP of the Podman host.'
    podman_host_ip = socket.gethostbyname(get_podman_hostname())
    display.info('Detected Podman host IP: %s' % podman_host_ip, verbosity=1)
    return podman_host_ip

@cache
def get_podman_default_hostname() -> t.Optional[str]:
    if False:
        return 10
    '\n    Return the default hostname of the Podman service.\n\n    --format was added in podman 3.3.0, this functionality depends on its availability\n    '
    hostname: t.Optional[str] = None
    try:
        stdout = raw_command(['podman', 'system', 'connection', 'list', '--format=json'], env=docker_environment(), capture=True)[0]
    except SubprocessError:
        stdout = '[]'
    try:
        connections = json.loads(stdout)
    except json.decoder.JSONDecodeError:
        return hostname
    for connection in connections:
        if connection['Name'][-1] == '*':
            hostname = connection['URI']
            break
    return hostname

@cache
def get_podman_remote() -> t.Optional[str]:
    if False:
        i = 10
        return i + 15
    'Return the remote podman hostname, if any, otherwise return None.'
    hostname = None
    podman_host = os.environ.get('CONTAINER_HOST')
    if not podman_host:
        podman_host = get_podman_default_hostname()
    if podman_host and podman_host.startswith('ssh://'):
        try:
            hostname = urllib.parse.urlparse(podman_host).hostname
        except ValueError:
            display.warning('Could not parse podman URI "%s"' % podman_host)
        else:
            display.info('Detected Podman remote: %s' % hostname, verbosity=1)
    return hostname

@cache
def get_podman_hostname() -> str:
    if False:
        print('Hello World!')
    'Return the hostname of the Podman service.'
    hostname = get_podman_remote()
    if not hostname:
        hostname = 'localhost'
        display.info('Assuming Podman is available on localhost.', verbosity=1)
    return hostname

@cache
def get_docker_container_id() -> t.Optional[str]:
    if False:
        i = 10
        return i + 15
    'Return the current container ID if running in a container, otherwise return None.'
    mountinfo_path = pathlib.Path('/proc/self/mountinfo')
    container_id = None
    engine = None
    if mountinfo_path.is_file():
        mounts = MountEntry.loads(mountinfo_path.read_text())
        for mount in mounts:
            if str(mount.path) == '/etc/hostname':
                if (match := re.search('/(?P<id>[0-9a-f]{64})/userdata/hostname$', str(mount.root))):
                    container_id = match.group('id')
                    engine = 'Podman'
                    break
                if (match := re.search('/(?P<id>[0-9a-f]{64})/hostname$', str(mount.root))):
                    container_id = match.group('id')
                    engine = 'Docker'
                    break
    if container_id:
        display.info(f'Detected execution in {engine} container ID: {container_id}', verbosity=1)
    return container_id

def docker_pull(args: CommonConfig, image: str) -> None:
    if False:
        while True:
            i = 10
    '\n    Pull the specified image if it is not available.\n    Images without a tag or digest will not be pulled.\n    Retries up to 10 times if the pull fails.\n    A warning will be shown for any image with volumes defined.\n    Images will be pulled only once.\n    Concurrent pulls for the same image will block until the first completes.\n    '
    with named_lock(f'docker_pull:{image}') as first:
        if first:
            __docker_pull(args, image)

def __docker_pull(args: CommonConfig, image: str) -> None:
    if False:
        while True:
            i = 10
    'Internal implementation for docker_pull. Do not call directly.'
    if '@' not in image and ':' not in image:
        display.info('Skipping pull of image without tag or digest: %s' % image, verbosity=2)
        inspect = docker_image_inspect(args, image)
    elif (inspect := docker_image_inspect(args, image, always=True)):
        display.info('Skipping pull of existing image: %s' % image, verbosity=2)
    else:
        for _iteration in range(1, 10):
            try:
                docker_command(args, ['pull', image], capture=False)
                if (inspect := docker_image_inspect(args, image)) or args.explain:
                    break
                display.warning(f'Image "{image}" not found after pull completed. Waiting a few seconds before trying again.')
            except SubprocessError:
                display.warning(f'Failed to pull container image "{image}". Waiting a few seconds before trying again.')
                time.sleep(3)
        else:
            raise ApplicationError(f'Failed to pull container image "{image}".')
    if inspect and inspect.volumes:
        display.warning(f'''Image "{image}" contains {len(inspect.volumes)} volume(s): {', '.join(sorted(inspect.volumes))}\nThis may result in leaking anonymous volumes. It may also prevent the image from working on some hosts or container engines.\nThe image should be rebuilt without the use of the VOLUME instruction.''', unique=True)

def docker_cp_to(args: CommonConfig, container_id: str, src: str, dst: str) -> None:
    if False:
        while True:
            i = 10
    'Copy a file to the specified container.'
    docker_command(args, ['cp', src, '%s:%s' % (container_id, dst)], capture=True)

def docker_create(args: CommonConfig, image: str, options: list[str], cmd: list[str]=None) -> tuple[t.Optional[str], t.Optional[str]]:
    if False:
        i = 10
        return i + 15
    'Create a container using the given docker image.'
    return docker_command(args, ['create'] + options + [image] + cmd, capture=True)

def docker_run(args: CommonConfig, image: str, options: list[str], cmd: list[str]=None, data: t.Optional[str]=None) -> tuple[t.Optional[str], t.Optional[str]]:
    if False:
        print('Hello World!')
    'Run a container using the given docker image.'
    return docker_command(args, ['run'] + options + [image] + cmd, data=data, capture=True)

def docker_start(args: CommonConfig, container_id: str, options: list[str]) -> tuple[t.Optional[str], t.Optional[str]]:
    if False:
        return 10
    'Start a container by name or ID.'
    return docker_command(args, ['start'] + options + [container_id], capture=True)

def docker_rm(args: CommonConfig, container_id: str) -> None:
    if False:
        i = 10
        return i + 15
    'Remove the specified container.'
    try:
        docker_command(args, ['stop', '--time', '0', container_id], capture=True)
        docker_command(args, ['rm', container_id], capture=True)
    except SubprocessError as ex:
        if 'no such container' not in ex.stderr.lower():
            raise ex

class DockerError(Exception):
    """General Docker error."""

class ContainerNotFoundError(DockerError):
    """The container identified by `identifier` was not found."""

    def __init__(self, identifier: str) -> None:
        if False:
            print('Hello World!')
        super().__init__('The container "%s" was not found.' % identifier)
        self.identifier = identifier

class DockerInspect:
    """The results of `docker inspect` for a single container."""

    def __init__(self, args: CommonConfig, inspection: dict[str, t.Any]) -> None:
        if False:
            print('Hello World!')
        self.args = args
        self.inspection = inspection

    @property
    def id(self) -> str:
        if False:
            print('Hello World!')
        'Return the ID of the container.'
        return self.inspection['Id']

    @property
    def network_settings(self) -> dict[str, t.Any]:
        if False:
            i = 10
            return i + 15
        'Return a dictionary of the container network settings.'
        return self.inspection['NetworkSettings']

    @property
    def state(self) -> dict[str, t.Any]:
        if False:
            for i in range(10):
                print('nop')
        'Return a dictionary of the container state.'
        return self.inspection['State']

    @property
    def config(self) -> dict[str, t.Any]:
        if False:
            return 10
        'Return a dictionary of the container configuration.'
        return self.inspection['Config']

    @property
    def ports(self) -> dict[str, list[dict[str, str]]]:
        if False:
            print('Hello World!')
        'Return a dictionary of ports the container has published.'
        return self.network_settings['Ports']

    @property
    def networks(self) -> t.Optional[dict[str, dict[str, t.Any]]]:
        if False:
            i = 10
            return i + 15
        'Return a dictionary of the networks the container is attached to, or None if running under podman, which does not support networks.'
        return self.network_settings.get('Networks')

    @property
    def running(self) -> bool:
        if False:
            print('Hello World!')
        'Return True if the container is running, otherwise False.'
        return self.state['Running']

    @property
    def pid(self) -> int:
        if False:
            while True:
                i = 10
        'Return the PID of the init process.'
        if self.args.explain:
            return 0
        return self.state['Pid']

    @property
    def env(self) -> list[str]:
        if False:
            while True:
                i = 10
        'Return a list of the environment variables used to create the container.'
        return self.config['Env']

    @property
    def image(self) -> str:
        if False:
            return 10
        'Return the image used to create the container.'
        return self.config['Image']

    def env_dict(self) -> dict[str, str]:
        if False:
            print('Hello World!')
        'Return a dictionary of the environment variables used to create the container.'
        return dict(((item[0], item[1]) for item in [e.split('=', 1) for e in self.env]))

    def get_tcp_port(self, port: int) -> t.Optional[list[dict[str, str]]]:
        if False:
            print('Hello World!')
        'Return a list of the endpoints published by the container for the specified TCP port, or None if it is not published.'
        return self.ports.get('%d/tcp' % port)

    def get_network_names(self) -> t.Optional[list[str]]:
        if False:
            return 10
        'Return a list of the network names the container is attached to.'
        if self.networks is None:
            return None
        return sorted(self.networks)

    def get_network_name(self) -> str:
        if False:
            i = 10
            return i + 15
        'Return the network name the container is attached to. Raises an exception if no network, or more than one, is attached.'
        networks = self.get_network_names()
        if not networks:
            raise ApplicationError('No network found for Docker container: %s.' % self.id)
        if len(networks) > 1:
            raise ApplicationError('Found multiple networks for Docker container %s instead of only one: %s' % (self.id, ', '.join(networks)))
        return networks[0]

def docker_inspect(args: CommonConfig, identifier: str, always: bool=False) -> DockerInspect:
    if False:
        i = 10
        return i + 15
    '\n    Return the results of `docker container inspect` for the specified container.\n    Raises a ContainerNotFoundError if the container was not found.\n    '
    try:
        stdout = docker_command(args, ['container', 'inspect', identifier], capture=True, always=always)[0]
    except SubprocessError as ex:
        stdout = ex.stdout
    if args.explain and (not always):
        items = []
    else:
        items = json.loads(stdout)
    if len(items) == 1:
        return DockerInspect(args, items[0])
    raise ContainerNotFoundError(identifier)

def docker_network_disconnect(args: CommonConfig, container_id: str, network: str) -> None:
    if False:
        i = 10
        return i + 15
    'Disconnect the specified docker container from the given network.'
    docker_command(args, ['network', 'disconnect', network, container_id], capture=True)

class DockerImageInspect:
    """The results of `docker image inspect` for a single image."""

    def __init__(self, args: CommonConfig, inspection: dict[str, t.Any]) -> None:
        if False:
            while True:
                i = 10
        self.args = args
        self.inspection = inspection

    @property
    def config(self) -> dict[str, t.Any]:
        if False:
            for i in range(10):
                print('nop')
        'Return a dictionary of the image config.'
        return self.inspection['Config']

    @property
    def volumes(self) -> dict[str, t.Any]:
        if False:
            for i in range(10):
                print('nop')
        'Return a dictionary of the image volumes.'
        return self.config.get('Volumes') or {}

    @property
    def cmd(self) -> list[str]:
        if False:
            return 10
        'The command to run when the container starts.'
        return self.config['Cmd']

@mutex
def docker_image_inspect(args: CommonConfig, image: str, always: bool=False) -> t.Optional[DockerImageInspect]:
    if False:
        i = 10
        return i + 15
    '\n    Return the results of `docker image inspect` for the specified image or None if the image does not exist.\n    '
    inspect_cache: dict[str, DockerImageInspect]
    try:
        inspect_cache = docker_image_inspect.cache
    except AttributeError:
        inspect_cache = docker_image_inspect.cache = {}
    if (inspect_result := inspect_cache.get(image)):
        return inspect_result
    try:
        stdout = docker_command(args, ['image', 'inspect', image], capture=True, always=always)[0]
    except SubprocessError:
        stdout = '[]'
    if args.explain and (not always):
        items = []
    else:
        items = json.loads(stdout)
    if len(items) > 1:
        raise ApplicationError(f'Inspection of image "{image}" resulted in {len(items)} items:\n{json.dumps(items, indent=4)}')
    if len(items) == 1:
        inspect_result = DockerImageInspect(args, items[0])
        inspect_cache[image] = inspect_result
        return inspect_result
    return None

class DockerNetworkInspect:
    """The results of `docker network inspect` for a single network."""

    def __init__(self, args: CommonConfig, inspection: dict[str, t.Any]) -> None:
        if False:
            i = 10
            return i + 15
        self.args = args
        self.inspection = inspection

def docker_network_inspect(args: CommonConfig, network: str, always: bool=False) -> t.Optional[DockerNetworkInspect]:
    if False:
        print('Hello World!')
    '\n    Return the results of `docker network inspect` for the specified network or None if the network does not exist.\n    '
    try:
        stdout = docker_command(args, ['network', 'inspect', network], capture=True, always=always)[0]
    except SubprocessError:
        stdout = '[]'
    if args.explain and (not always):
        items = []
    else:
        items = json.loads(stdout)
    if len(items) == 1:
        return DockerNetworkInspect(args, items[0])
    return None

def docker_logs(args: CommonConfig, container_id: str) -> None:
    if False:
        print('Hello World!')
    'Display logs for the specified container. If an error occurs, it is displayed rather than raising an exception.'
    try:
        docker_command(args, ['logs', container_id], capture=False)
    except SubprocessError as ex:
        display.error(str(ex))

def docker_exec(args: CommonConfig, container_id: str, cmd: list[str], capture: bool, options: t.Optional[list[str]]=None, stdin: t.Optional[t.IO[bytes]]=None, stdout: t.Optional[t.IO[bytes]]=None, interactive: bool=False, output_stream: t.Optional[OutputStream]=None, data: t.Optional[str]=None) -> tuple[t.Optional[str], t.Optional[str]]:
    if False:
        for i in range(10):
            print('nop')
    'Execute the given command in the specified container.'
    if not options:
        options = []
    if data or stdin or stdout:
        options.append('-i')
    return docker_command(args, ['exec'] + options + [container_id] + cmd, capture=capture, stdin=stdin, stdout=stdout, interactive=interactive, output_stream=output_stream, data=data)

def docker_command(args: CommonConfig, cmd: list[str], capture: bool, stdin: t.Optional[t.IO[bytes]]=None, stdout: t.Optional[t.IO[bytes]]=None, interactive: bool=False, output_stream: t.Optional[OutputStream]=None, always: bool=False, data: t.Optional[str]=None) -> tuple[t.Optional[str], t.Optional[str]]:
    if False:
        i = 10
        return i + 15
    'Run the specified docker command.'
    env = docker_environment()
    command = [require_docker().command]
    if command[0] == 'podman' and get_podman_remote():
        command.append('--remote')
    return run_command(args, command + cmd, env=env, capture=capture, stdin=stdin, stdout=stdout, interactive=interactive, always=always, output_stream=output_stream, data=data)

def docker_environment() -> dict[str, str]:
    if False:
        while True:
            i = 10
    'Return a dictionary of docker related environment variables found in the current environment.'
    env = common_environment()
    var_names = {'XDG_RUNTIME_DIR'}
    var_prefixes = {'CONTAINER_', 'DOCKER_'}
    env.update({name: value for (name, value) in os.environ.items() if name in var_names or any((name.startswith(prefix) for prefix in var_prefixes))})
    return env