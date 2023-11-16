"""Profiles to represent individual test hosts or a user-provided inventory file."""
from __future__ import annotations
import abc
import dataclasses
import os
import shlex
import tempfile
import time
import typing as t
from .io import read_text_file, write_text_file
from .config import CommonConfig, EnvironmentConfig, IntegrationConfig, TerminateMode
from .host_configs import ControllerConfig, ControllerHostConfig, DockerConfig, HostConfig, NetworkInventoryConfig, NetworkRemoteConfig, OriginConfig, PosixConfig, PosixRemoteConfig, PosixSshConfig, PythonConfig, RemoteConfig, VirtualPythonConfig, WindowsInventoryConfig, WindowsRemoteConfig
from .core_ci import AnsibleCoreCI, SshKey, VmResource
from .util import ApplicationError, SubprocessError, cache, display, get_type_map, sanitize_host_name, sorted_versions, InternalError, HostConnectionError, ANSIBLE_TEST_TARGET_ROOT
from .util_common import get_docs_url, intercept_python
from .docker_util import docker_exec, docker_image_inspect, docker_logs, docker_pull, docker_rm, get_docker_hostname, require_docker, get_docker_info, detect_host_properties, run_utility_container, SystemdControlGroupV1Status, LOGINUID_NOT_SET, UTILITY_IMAGE
from .bootstrap import BootstrapDocker, BootstrapRemote
from .venv import get_virtual_python
from .ssh import SshConnectionDetail
from .ansible_util import ansible_environment, get_hosts, parse_inventory
from .containers import HostType, get_container_database, run_support_container
from .connections import Connection, DockerConnection, LocalConnection, SshConnection
from .become import Become, SUPPORTED_BECOME_METHODS, Sudo
from .completion import AuditMode, CGroupVersion
from .dev.container_probe import CGroupMount, CGroupPath, CGroupState, MountType, check_container_cgroup_status
TControllerHostConfig = t.TypeVar('TControllerHostConfig', bound=ControllerHostConfig)
THostConfig = t.TypeVar('THostConfig', bound=HostConfig)
TPosixConfig = t.TypeVar('TPosixConfig', bound=PosixConfig)
TRemoteConfig = t.TypeVar('TRemoteConfig', bound=RemoteConfig)

class ControlGroupError(ApplicationError):
    """Raised when the container host does not have the necessary cgroup support to run a container."""

    def __init__(self, args: CommonConfig, reason: str) -> None:
        if False:
            while True:
                i = 10
        engine = require_docker().command
        dd_wsl2 = get_docker_info(args).docker_desktop_wsl2
        message = f'\n{reason}\n\nRun the following commands as root on the container host to resolve this issue:\n\n  mkdir /sys/fs/cgroup/systemd\n  mount cgroup -t cgroup /sys/fs/cgroup/systemd -o none,name=systemd,xattr\n  chown -R {{user}}:{{group}} /sys/fs/cgroup/systemd  # only when rootless\n\nNOTE: These changes must be applied each time the container host is rebooted.\n'.strip()
        podman_message = "\n      If rootless Podman is already running [1], you may need to stop it before\n      containers are able to use the new mount point.\n\n[1] Check for 'podman' and 'catatonit' processes.\n"
        dd_wsl_message = f"\n      When using Docker Desktop with WSL2, additional configuration [1] is required.\n\n[1] {get_docs_url('https://docs.ansible.com/ansible-core/devel/dev_guide/testing_running_locally.html#docker-desktop-with-wsl2')}\n"
        if engine == 'podman':
            message += podman_message
        elif dd_wsl2:
            message += dd_wsl_message
        message = message.strip()
        super().__init__(message)

@dataclasses.dataclass(frozen=True)
class Inventory:
    """Simple representation of an Ansible inventory."""
    host_groups: dict[str, dict[str, dict[str, t.Union[str, int]]]]
    extra_groups: t.Optional[dict[str, list[str]]] = None

    @staticmethod
    def create_single_host(name: str, variables: dict[str, t.Union[str, int]]) -> Inventory:
        if False:
            return 10
        'Return an inventory instance created from the given hostname and variables.'
        return Inventory(host_groups=dict(all={name: variables}))

    def write(self, args: CommonConfig, path: str) -> None:
        if False:
            while True:
                i = 10
        'Write the given inventory to the specified path on disk.'
        inventory_text = ''
        for (group, hosts) in self.host_groups.items():
            inventory_text += f'[{group}]\n'
            for (host, variables) in hosts.items():
                kvp = ' '.join((f'{key}="{value}"' for (key, value) in variables.items()))
                inventory_text += f'{host} {kvp}\n'
            inventory_text += '\n'
        for (group, children) in (self.extra_groups or {}).items():
            inventory_text += f'[{group}]\n'
            for child in children:
                inventory_text += f'{child}\n'
            inventory_text += '\n'
        inventory_text = inventory_text.strip()
        if not args.explain:
            write_text_file(path, inventory_text + '\n')
        display.info(f'>>> Inventory\n{inventory_text}', verbosity=3)

class HostProfile(t.Generic[THostConfig], metaclass=abc.ABCMeta):
    """Base class for host profiles."""

    def __init__(self, *, args: EnvironmentConfig, config: THostConfig, targets: t.Optional[list[HostConfig]]) -> None:
        if False:
            print('Hello World!')
        self.args = args
        self.config = config
        self.controller = bool(targets)
        self.targets = targets or []
        self.state: dict[str, t.Any] = {}
        'State that must be persisted across delegation.'
        self.cache: dict[str, t.Any] = {}
        'Cache that must not be persisted across delegation.'

    def provision(self) -> None:
        if False:
            while True:
                i = 10
        'Provision the host before delegation.'

    def setup(self) -> None:
        if False:
            i = 10
            return i + 15
        'Perform out-of-band setup before delegation.'

    def on_target_failure(self) -> None:
        if False:
            i = 10
            return i + 15
        'Executed during failure handling if this profile is a target.'

    def deprovision(self) -> None:
        if False:
            print('Hello World!')
        'Deprovision the host after delegation has completed.'

    def wait(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Wait for the instance to be ready. Executed before delegation for the controller and after delegation for targets.'

    def configure(self) -> None:
        if False:
            while True:
                i = 10
        'Perform in-band configuration. Executed before delegation for the controller and after delegation for targets.'

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        return {key: value for (key, value) in self.__dict__.items() if key not in ('args', 'cache')}

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        self.__dict__.update(state)
        self.cache = {}

class PosixProfile(HostProfile[TPosixConfig], metaclass=abc.ABCMeta):
    """Base class for POSIX host profiles."""

    @property
    def python(self) -> PythonConfig:
        if False:
            for i in range(10):
                print('nop')
        '\n        The Python to use for this profile.\n        If it is a virtual python, it will be created the first time it is requested.\n        '
        python = self.state.get('python')
        if not python:
            python = self.config.python
            if isinstance(python, VirtualPythonConfig):
                python = get_virtual_python(self.args, python)
            self.state['python'] = python
        return python

class ControllerHostProfile(PosixProfile[TControllerHostConfig], metaclass=abc.ABCMeta):
    """Base class for profiles usable as a controller."""

    @abc.abstractmethod
    def get_origin_controller_connection(self) -> Connection:
        if False:
            while True:
                i = 10
        'Return a connection for accessing the host as a controller from the origin.'

    @abc.abstractmethod
    def get_working_directory(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return the working directory for the host.'

class SshTargetHostProfile(HostProfile[THostConfig], metaclass=abc.ABCMeta):
    """Base class for profiles offering SSH connectivity."""

    @abc.abstractmethod
    def get_controller_target_connections(self) -> list[SshConnection]:
        if False:
            return 10
        'Return SSH connection(s) for accessing the host as a target from the controller.'

class RemoteProfile(SshTargetHostProfile[TRemoteConfig], metaclass=abc.ABCMeta):
    """Base class for remote instance profiles."""

    @property
    def core_ci_state(self) -> t.Optional[dict[str, str]]:
        if False:
            return 10
        'The saved Ansible Core CI state.'
        return self.state.get('core_ci')

    @core_ci_state.setter
    def core_ci_state(self, value: dict[str, str]) -> None:
        if False:
            while True:
                i = 10
        'The saved Ansible Core CI state.'
        self.state['core_ci'] = value

    def provision(self) -> None:
        if False:
            i = 10
            return i + 15
        'Provision the host before delegation.'
        self.core_ci = self.create_core_ci(load=True)
        self.core_ci.start()
        self.core_ci_state = self.core_ci.save()

    def deprovision(self) -> None:
        if False:
            i = 10
            return i + 15
        'Deprovision the host after delegation has completed.'
        if self.args.remote_terminate == TerminateMode.ALWAYS or (self.args.remote_terminate == TerminateMode.SUCCESS and self.args.success):
            self.delete_instance()

    @property
    def core_ci(self) -> t.Optional[AnsibleCoreCI]:
        if False:
            i = 10
            return i + 15
        'Return the cached AnsibleCoreCI instance, if any, otherwise None.'
        return self.cache.get('core_ci')

    @core_ci.setter
    def core_ci(self, value: AnsibleCoreCI) -> None:
        if False:
            while True:
                i = 10
        'Cache the given AnsibleCoreCI instance.'
        self.cache['core_ci'] = value

    def get_instance(self) -> t.Optional[AnsibleCoreCI]:
        if False:
            print('Hello World!')
        'Return the current AnsibleCoreCI instance, loading it if not already loaded.'
        if not self.core_ci and self.core_ci_state:
            self.core_ci = self.create_core_ci(load=False)
            self.core_ci.load(self.core_ci_state)
        return self.core_ci

    def delete_instance(self) -> None:
        if False:
            return 10
        'Delete the AnsibleCoreCI VM instance.'
        core_ci = self.get_instance()
        if not core_ci:
            return
        core_ci.stop()

    def wait_for_instance(self) -> AnsibleCoreCI:
        if False:
            for i in range(10):
                print('nop')
        'Wait for an AnsibleCoreCI VM instance to become ready.'
        core_ci = self.get_instance()
        core_ci.wait()
        return core_ci

    def create_core_ci(self, load: bool) -> AnsibleCoreCI:
        if False:
            print('Hello World!')
        'Create and return an AnsibleCoreCI instance.'
        if not self.config.arch:
            raise InternalError(f'No arch specified for config: {self.config}')
        return AnsibleCoreCI(args=self.args, resource=VmResource(platform=self.config.platform, version=self.config.version, architecture=self.config.arch, provider=self.config.provider, tag='controller' if self.controller else 'target'), load=load)

class ControllerProfile(SshTargetHostProfile[ControllerConfig], PosixProfile[ControllerConfig]):
    """Host profile for the controller as a target."""

    def get_controller_target_connections(self) -> list[SshConnection]:
        if False:
            i = 10
            return i + 15
        'Return SSH connection(s) for accessing the host as a target from the controller.'
        settings = SshConnectionDetail(name='localhost', host='localhost', port=None, user='root', identity_file=SshKey(self.args).key, python_interpreter=self.args.controller_python.path)
        return [SshConnection(self.args, settings)]

class DockerProfile(ControllerHostProfile[DockerConfig], SshTargetHostProfile[DockerConfig]):
    """Host profile for a docker instance."""
    MARKER = 'ansible-test-marker'

    @dataclasses.dataclass(frozen=True)
    class InitConfig:
        """Configuration details required to run the container init."""
        options: list[str]
        command: str
        command_privileged: bool
        expected_mounts: tuple[CGroupMount, ...]

    @property
    def container_name(self) -> t.Optional[str]:
        if False:
            i = 10
            return i + 15
        'Return the stored container name, if any, otherwise None.'
        return self.state.get('container_name')

    @container_name.setter
    def container_name(self, value: str) -> None:
        if False:
            print('Hello World!')
        'Store the given container name.'
        self.state['container_name'] = value

    @property
    def cgroup_path(self) -> t.Optional[str]:
        if False:
            i = 10
            return i + 15
        'Return the path to the cgroup v1 systemd hierarchy, if any, otherwise None.'
        return self.state.get('cgroup_path')

    @cgroup_path.setter
    def cgroup_path(self, value: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Store the path to the cgroup v1 systemd hierarchy.'
        self.state['cgroup_path'] = value

    @property
    def label(self) -> str:
        if False:
            while True:
                i = 10
        'Label to apply to resources related to this profile.'
        return f"{('controller' if self.controller else 'target')}"

    def provision(self) -> None:
        if False:
            while True:
                i = 10
        'Provision the host before delegation.'
        init_probe = self.args.dev_probe_cgroups is not None
        init_config = self.get_init_config()
        container = run_support_container(args=self.args, context='__test_hosts__', image=self.config.image, name=f'ansible-test-{self.label}', ports=[22], publish_ports=not self.controller, options=init_config.options, cleanup=False, cmd=self.build_init_command(init_config, init_probe))
        if not container:
            if self.args.prime_containers:
                if init_config.command_privileged or init_probe:
                    docker_pull(self.args, UTILITY_IMAGE)
            return
        self.container_name = container.name
        try:
            options = ['--pid', 'host', '--privileged']
            if init_config.command and init_config.command_privileged:
                init_command = init_config.command
                if not init_probe:
                    init_command += f' && {shlex.join(self.wake_command)}'
                cmd = ['nsenter', '-t', str(container.details.container.pid), '-m', '-p', 'sh', '-c', init_command]
                run_utility_container(self.args, f'ansible-test-init-{self.label}', cmd, options)
            if init_probe:
                check_container_cgroup_status(self.args, self.config, self.container_name, init_config.expected_mounts)
                cmd = ['nsenter', '-t', str(container.details.container.pid), '-m', '-p'] + self.wake_command
                run_utility_container(self.args, f'ansible-test-wake-{self.label}', cmd, options)
        except SubprocessError:
            display.info(f'Checking container "{self.container_name}" logs...')
            docker_logs(self.args, self.container_name)
            raise

    def get_init_config(self) -> InitConfig:
        if False:
            while True:
                i = 10
        'Return init config for running under the current container engine.'
        self.check_cgroup_requirements()
        engine = require_docker().command
        init_config = getattr(self, f'get_{engine}_init_config')()
        return init_config

    def get_podman_init_config(self) -> InitConfig:
        if False:
            return 10
        'Return init config for running under Podman.'
        options = self.get_common_run_options()
        command: t.Optional[str] = None
        command_privileged = False
        expected_mounts: tuple[CGroupMount, ...]
        cgroup_version = get_docker_info(self.args).cgroup_version
        options.extend(('--cap-add', 'SYS_CHROOT'))
        if self.config.audit == AuditMode.REQUIRED and detect_host_properties(self.args).audit_code == 'EPERM':
            options.extend(('--cap-add', 'AUDIT_WRITE'))
        if (loginuid := detect_host_properties(self.args).loginuid) not in (0, LOGINUID_NOT_SET, None):
            display.warning(f'Running containers with capability AUDIT_CONTROL since the container loginuid ({loginuid}) is incorrect. This is most likely due to use of sudo to run ansible-test when loginuid is already set.', unique=True)
            options.extend(('--cap-add', 'AUDIT_CONTROL'))
        if self.config.cgroup == CGroupVersion.NONE:
            options.extend(('--systemd', 'false', '--cgroupns', 'private', '--tmpfs', '/sys/fs/cgroup'))
            expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.TMPFS, writable=True, state=None),)
        elif self.config.cgroup in (CGroupVersion.V1_V2, CGroupVersion.V1_ONLY) and cgroup_version == 1:
            options.extend(('--systemd', 'always', '--cgroupns', 'host', '--tmpfs', '/sys/fs/cgroup'))
            self.check_systemd_cgroup_v1(options)
            expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.TMPFS, writable=True, state=None), CGroupMount(path=CGroupPath.SYSTEMD, type=MountType.CGROUP_V1, writable=None, state=CGroupState.HOST), CGroupMount(path=CGroupPath.SYSTEMD_RELEASE_AGENT, type=None, writable=False, state=None))
        elif self.config.cgroup in (CGroupVersion.V1_V2, CGroupVersion.V2_ONLY) and cgroup_version == 2:
            options.extend(('--systemd', 'always', '--cgroupns', 'private'))
            expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.CGROUP_V2, writable=True, state=CGroupState.PRIVATE),)
        elif self.config.cgroup == CGroupVersion.V1_ONLY and cgroup_version == 2:
            cgroup_path = self.create_systemd_cgroup_v1()
            command = f'echo 1 > {cgroup_path}/cgroup.procs'
            options.extend(('--systemd', 'always', '--cgroupns', 'private', '--volume', '/sys/fs/cgroup/systemd:/sys/fs/cgroup/systemd:ro', '--volume', f'{cgroup_path}:{cgroup_path}:rw'))
            expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.CGROUP_V2, writable=True, state=CGroupState.PRIVATE), CGroupMount(path=CGroupPath.SYSTEMD, type=MountType.CGROUP_V1, writable=False, state=CGroupState.SHADOWED), CGroupMount(path=cgroup_path, type=MountType.CGROUP_V1, writable=True, state=CGroupState.HOST))
        else:
            raise InternalError(f'Unhandled cgroup configuration: {self.config.cgroup} on cgroup v{cgroup_version}.')
        return self.InitConfig(options=options, command=command, command_privileged=command_privileged, expected_mounts=expected_mounts)

    def get_docker_init_config(self) -> InitConfig:
        if False:
            while True:
                i = 10
        'Return init config for running under Docker.'
        options = self.get_common_run_options()
        command: t.Optional[str] = None
        command_privileged = False
        expected_mounts: tuple[CGroupMount, ...]
        cgroup_version = get_docker_info(self.args).cgroup_version
        if self.config.cgroup == CGroupVersion.NONE:
            if get_docker_info(self.args).cgroupns_option_supported:
                options.extend(('--cgroupns', 'private'))
            options.extend(('--tmpfs', '/sys/fs/cgroup'))
            expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.TMPFS, writable=True, state=None),)
        elif self.config.cgroup in (CGroupVersion.V1_V2, CGroupVersion.V1_ONLY) and cgroup_version == 1:
            if get_docker_info(self.args).cgroupns_option_supported:
                options.extend(('--cgroupns', 'host'))
            options.extend(('--tmpfs', '/sys/fs/cgroup', '--volume', '/sys/fs/cgroup/systemd:/sys/fs/cgroup/systemd:rw'))
            self.check_systemd_cgroup_v1(options)
            expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.TMPFS, writable=True, state=None), CGroupMount(path=CGroupPath.SYSTEMD, type=MountType.CGROUP_V1, writable=True, state=CGroupState.HOST))
        elif self.config.cgroup in (CGroupVersion.V1_V2, CGroupVersion.V2_ONLY) and cgroup_version == 2:
            command = 'mount -o remount,rw /sys/fs/cgroup/'
            command_privileged = True
            options.extend(('--cgroupns', 'private'))
            expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.CGROUP_V2, writable=True, state=CGroupState.PRIVATE),)
        elif self.config.cgroup == CGroupVersion.V1_ONLY and cgroup_version == 2:
            cgroup_path = self.create_systemd_cgroup_v1()
            command = f'echo 1 > {cgroup_path}/cgroup.procs'
            options.extend(('--cgroupns', 'private', '--tmpfs', '/sys/fs/cgroup', '--tmpfs', '/sys/fs/cgroup/systemd', '--volume', f'{cgroup_path}:{cgroup_path}:rw'))
            expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.TMPFS, writable=True, state=None), CGroupMount(path=CGroupPath.SYSTEMD, type=MountType.TMPFS, writable=True, state=None), CGroupMount(path=cgroup_path, type=MountType.CGROUP_V1, writable=True, state=CGroupState.HOST))
        else:
            raise InternalError(f'Unhandled cgroup configuration: {self.config.cgroup} on cgroup v{cgroup_version}.')
        return self.InitConfig(options=options, command=command, command_privileged=command_privileged, expected_mounts=expected_mounts)

    def build_init_command(self, init_config: InitConfig, sleep: bool) -> t.Optional[list[str]]:
        if False:
            i = 10
            return i + 15
        '\n        Build and return the command to start in the container.\n        Returns None if the default command for the container should be used.\n\n        The sleep duration below was selected to:\n\n          - Allow enough time to perform necessary operations in the container before waking it.\n          - Make the delay obvious if the wake command doesn\'t run or succeed.\n          - Avoid hanging indefinitely or for an unreasonably long time.\n\n        NOTE: The container must have a POSIX-compliant default shell "sh" with a non-builtin "sleep" command.\n              The "sleep" command is invoked through "env" to avoid using a shell builtin "sleep" (if present).\n        '
        command = ''
        if init_config.command and (not init_config.command_privileged):
            command += f'{init_config.command} && '
        if sleep or init_config.command_privileged:
            command += 'env sleep 60 ; '
        if not command:
            return None
        docker_pull(self.args, self.config.image)
        inspect = docker_image_inspect(self.args, self.config.image)
        command += f'exec {shlex.join(inspect.cmd)}'
        return ['sh', '-c', command]

    @property
    def wake_command(self) -> list[str]:
        if False:
            print('Hello World!')
        '\n        The command used to wake the container from sleep.\n        This will be run inside our utility container, so the command used does not need to be present in the container being woken up.\n        '
        return ['pkill', 'sleep']

    def check_systemd_cgroup_v1(self, options: list[str]) -> None:
        if False:
            print('Hello World!')
        'Check the cgroup v1 systemd hierarchy to verify it is writeable for our container.'
        probe_script = read_text_file(os.path.join(ANSIBLE_TEST_TARGET_ROOT, 'setup', 'check_systemd_cgroup_v1.sh')).replace('@MARKER@', self.MARKER).replace('@LABEL@', f'{self.label}-{self.args.session_name}')
        cmd = ['sh']
        try:
            run_utility_container(self.args, f'ansible-test-cgroup-check-{self.label}', cmd, options, data=probe_script)
        except SubprocessError as ex:
            if (error := self.extract_error(ex.stderr)):
                raise ControlGroupError(self.args, f'Unable to create a v1 cgroup within the systemd hierarchy.\nReason: {error}') from ex
            raise

    def create_systemd_cgroup_v1(self) -> str:
        if False:
            return 10
        'Create a unique ansible-test cgroup in the v1 systemd hierarchy and return its path.'
        self.cgroup_path = f'/sys/fs/cgroup/systemd/ansible-test-{self.label}-{self.args.session_name}'
        options = ['--volume', '/sys/fs/cgroup/systemd:/sys/fs/cgroup/systemd:rw', '--privileged']
        cmd = ['sh', '-c', f'>&2 echo {shlex.quote(self.MARKER)} && mkdir {shlex.quote(self.cgroup_path)}']
        try:
            run_utility_container(self.args, f'ansible-test-cgroup-create-{self.label}', cmd, options)
        except SubprocessError as ex:
            if (error := self.extract_error(ex.stderr)):
                raise ControlGroupError(self.args, f'Unable to create a v1 cgroup within the systemd hierarchy.\nReason: {error}') from ex
            raise
        return self.cgroup_path

    @property
    def delete_systemd_cgroup_v1_command(self) -> list[str]:
        if False:
            i = 10
            return i + 15
        'The command used to remove the previously created ansible-test cgroup in the v1 systemd hierarchy.'
        return ['find', self.cgroup_path, '-type', 'd', '-delete']

    def delete_systemd_cgroup_v1(self) -> None:
        if False:
            print('Hello World!')
        'Delete a previously created ansible-test cgroup in the v1 systemd hierarchy.'
        options = ['--volume', '/sys/fs/cgroup/systemd:/sys/fs/cgroup/systemd:rw', '--privileged']
        cmd = ['sh', '-c', f'>&2 echo {shlex.quote(self.MARKER)} && {shlex.join(self.delete_systemd_cgroup_v1_command)}']
        try:
            run_utility_container(self.args, f'ansible-test-cgroup-delete-{self.label}', cmd, options)
        except SubprocessError as ex:
            if (error := self.extract_error(ex.stderr)):
                if error.endswith(': No such file or directory'):
                    return
            display.error(str(ex))

    def extract_error(self, value: str) -> t.Optional[str]:
        if False:
            return 10
        '\n        Extract the ansible-test portion of the error message from the given value and return it.\n        Returns None if no ansible-test marker was found.\n        '
        lines = value.strip().splitlines()
        try:
            idx = lines.index(self.MARKER)
        except ValueError:
            return None
        lines = lines[idx + 1:]
        message = '\n'.join(lines)
        return message

    def check_cgroup_requirements(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Check cgroup requirements for the container.'
        cgroup_version = get_docker_info(self.args).cgroup_version
        if cgroup_version not in (1, 2):
            raise ApplicationError(f'The container host provides cgroup v{cgroup_version}, but only version v1 and v2 are supported.')
        if self.config.cgroup == CGroupVersion.V2_ONLY and cgroup_version != 2:
            raise ApplicationError(f'Container {self.config.name} requires cgroup v2 but the container host provides cgroup v{cgroup_version}.')
        if self.config.cgroup == CGroupVersion.V1_ONLY or (self.config.cgroup != CGroupVersion.NONE and get_docker_info(self.args).cgroup_version == 1):
            if (cgroup_v1 := detect_host_properties(self.args).cgroup_v1) != SystemdControlGroupV1Status.VALID:
                if self.config.cgroup == CGroupVersion.V1_ONLY:
                    if get_docker_info(self.args).cgroup_version == 2:
                        reason = f'Container {self.config.name} requires cgroup v1, but the container host only provides cgroup v2.'
                    else:
                        reason = f'Container {self.config.name} requires cgroup v1, but the container host does not appear to be running systemd.'
                else:
                    reason = 'The container host provides cgroup v1, but does not appear to be running systemd.'
                reason += f'\n{cgroup_v1.value}'
                raise ControlGroupError(self.args, reason)

    def setup(self) -> None:
        if False:
            while True:
                i = 10
        'Perform out-of-band setup before delegation.'
        bootstrapper = BootstrapDocker(controller=self.controller, python_versions=[self.python.version], ssh_key=SshKey(self.args))
        setup_sh = bootstrapper.get_script()
        shell = setup_sh.splitlines()[0][2:]
        try:
            docker_exec(self.args, self.container_name, [shell], data=setup_sh, capture=False)
        except SubprocessError:
            display.info(f'Checking container "{self.container_name}" logs...')
            docker_logs(self.args, self.container_name)
            raise

    def deprovision(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Deprovision the host after delegation has completed.'
        container_exists = False
        if self.container_name:
            if self.args.docker_terminate == TerminateMode.ALWAYS or (self.args.docker_terminate == TerminateMode.SUCCESS and self.args.success):
                docker_rm(self.args, self.container_name)
            else:
                container_exists = True
        if self.cgroup_path:
            if container_exists:
                display.notice(f'Remember to run `{require_docker().command} rm -f {self.container_name}` when finished testing. Then run `{shlex.join(self.delete_systemd_cgroup_v1_command)}` on the container host.')
            else:
                self.delete_systemd_cgroup_v1()
        elif container_exists:
            display.notice(f'Remember to run `{require_docker().command} rm -f {self.container_name}` when finished testing.')

    def wait(self) -> None:
        if False:
            while True:
                i = 10
        'Wait for the instance to be ready. Executed before delegation for the controller and after delegation for targets.'
        if not self.controller:
            con = self.get_controller_target_connections()[0]
            last_error = ''
            for dummy in range(1, 10):
                try:
                    con.run(['id'], capture=True)
                except SubprocessError as ex:
                    if 'Permission denied' in ex.message:
                        raise
                    last_error = str(ex)
                    time.sleep(1)
                else:
                    return
            display.info('Checking SSH debug output...')
            display.info(last_error)
            if not self.args.delegate and (not self.args.host_path):

                def callback() -> None:
                    if False:
                        for i in range(10):
                            print('nop')
                    'Callback to run during error display.'
                    self.on_target_failure()
            else:
                callback = None
            raise HostConnectionError(f'Timeout waiting for {self.config.name} container {self.container_name}.', callback)

    def get_controller_target_connections(self) -> list[SshConnection]:
        if False:
            for i in range(10):
                print('nop')
        'Return SSH connection(s) for accessing the host as a target from the controller.'
        containers = get_container_database(self.args)
        access = containers.data[HostType.control]['__test_hosts__'][self.container_name]
        host = access.host_ip
        port = dict(access.port_map())[22]
        settings = SshConnectionDetail(name=self.config.name, user='root', host=host, port=port, identity_file=SshKey(self.args).key, python_interpreter=self.python.path, enable_rsa_sha1='centos6' in self.config.image)
        return [SshConnection(self.args, settings)]

    def get_origin_controller_connection(self) -> DockerConnection:
        if False:
            while True:
                i = 10
        'Return a connection for accessing the host as a controller from the origin.'
        return DockerConnection(self.args, self.container_name)

    def get_working_directory(self) -> str:
        if False:
            while True:
                i = 10
        'Return the working directory for the host.'
        return '/root'

    def on_target_failure(self) -> None:
        if False:
            return 10
        'Executed during failure handling if this profile is a target.'
        display.info(f'Checking container "{self.container_name}" logs...')
        try:
            docker_logs(self.args, self.container_name)
        except SubprocessError as ex:
            display.error(str(ex))
        if self.config.cgroup != CGroupVersion.NONE:
            display.info(f'Checking container "{self.container_name}" systemd logs...')
            try:
                docker_exec(self.args, self.container_name, ['journalctl'], capture=False)
            except SubprocessError as ex:
                display.error(str(ex))
        display.error(f'Connection to container "{self.container_name}" failed. See logs and original error above.')

    def get_common_run_options(self) -> list[str]:
        if False:
            i = 10
            return i + 15
        'Return a list of options needed to run the container.'
        options = ['--tmpfs', '/tmp:exec', '--tmpfs', '/run:exec', '--tmpfs', '/run/lock']
        if self.config.privileged:
            options.append('--privileged')
        if self.config.memory:
            options.extend([f'--memory={self.config.memory}', f'--memory-swap={self.config.memory}'])
        if self.config.seccomp != 'default':
            options.extend(['--security-opt', f'seccomp={self.config.seccomp}'])
        docker_socket = '/var/run/docker.sock'
        if get_docker_hostname() != 'localhost' or os.path.exists(docker_socket):
            options.extend(['--volume', f'{docker_socket}:{docker_socket}'])
        return options

class NetworkInventoryProfile(HostProfile[NetworkInventoryConfig]):
    """Host profile for a network inventory."""

class NetworkRemoteProfile(RemoteProfile[NetworkRemoteConfig]):
    """Host profile for a network remote instance."""

    def wait(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Wait for the instance to be ready. Executed before delegation for the controller and after delegation for targets.'
        self.wait_until_ready()

    def get_inventory_variables(self) -> dict[str, t.Optional[t.Union[str, int]]]:
        if False:
            return 10
        'Return inventory variables for accessing this host.'
        core_ci = self.wait_for_instance()
        connection = core_ci.connection
        variables: dict[str, t.Optional[t.Union[str, int]]] = dict(ansible_connection=self.config.connection, ansible_pipelining='yes', ansible_host=connection.hostname, ansible_port=connection.port, ansible_user=connection.username, ansible_ssh_private_key_file=core_ci.ssh_key.key, ansible_paramiko_use_rsa_sha2_algorithms='no', ansible_network_os=f'{self.config.collection}.{self.config.platform}' if self.config.collection else self.config.platform)
        return variables

    def wait_until_ready(self) -> None:
        if False:
            while True:
                i = 10
        'Wait for the host to respond to an Ansible module request.'
        core_ci = self.wait_for_instance()
        if not isinstance(self.args, IntegrationConfig):
            return
        inventory = Inventory.create_single_host(sanitize_host_name(self.config.name), self.get_inventory_variables())
        env = ansible_environment(self.args)
        module_name = f"{(self.config.collection + '.' if self.config.collection else '')}{self.config.platform}_command"
        with tempfile.NamedTemporaryFile() as inventory_file:
            inventory.write(self.args, inventory_file.name)
            cmd = ['ansible', '-m', module_name, '-a', 'commands=?', '-i', inventory_file.name, 'all']
            for dummy in range(1, 90):
                try:
                    intercept_python(self.args, self.args.controller_python, cmd, env, capture=True)
                except SubprocessError as ex:
                    display.warning(str(ex))
                    time.sleep(10)
                else:
                    return
            raise HostConnectionError(f'Timeout waiting for {self.config.name} instance {core_ci.instance_id}.')

    def get_controller_target_connections(self) -> list[SshConnection]:
        if False:
            print('Hello World!')
        'Return SSH connection(s) for accessing the host as a target from the controller.'
        core_ci = self.wait_for_instance()
        settings = SshConnectionDetail(name=core_ci.name, host=core_ci.connection.hostname, port=core_ci.connection.port, user=core_ci.connection.username, identity_file=core_ci.ssh_key.key, enable_rsa_sha1=True)
        return [SshConnection(self.args, settings)]

class OriginProfile(ControllerHostProfile[OriginConfig]):
    """Host profile for origin."""

    def get_origin_controller_connection(self) -> LocalConnection:
        if False:
            while True:
                i = 10
        'Return a connection for accessing the host as a controller from the origin.'
        return LocalConnection(self.args)

    def get_working_directory(self) -> str:
        if False:
            return 10
        'Return the working directory for the host.'
        return os.getcwd()

class PosixRemoteProfile(ControllerHostProfile[PosixRemoteConfig], RemoteProfile[PosixRemoteConfig]):
    """Host profile for a POSIX remote instance."""

    def wait(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Wait for the instance to be ready. Executed before delegation for the controller and after delegation for targets.'
        self.wait_until_ready()

    def configure(self) -> None:
        if False:
            print('Hello World!')
        'Perform in-band configuration. Executed before delegation for the controller and after delegation for targets.'
        python_versions = [self.python.version] + [target.python.version for target in self.targets if isinstance(target, ControllerConfig)]
        python_versions = sorted_versions(list(set(python_versions)))
        core_ci = self.wait_for_instance()
        pwd = self.wait_until_ready()
        display.info(f'Remote working directory: {pwd}', verbosity=1)
        bootstrapper = BootstrapRemote(controller=self.controller, platform=self.config.platform, platform_version=self.config.version, python_versions=python_versions, ssh_key=core_ci.ssh_key)
        setup_sh = bootstrapper.get_script()
        shell = setup_sh.splitlines()[0][2:]
        ssh = self.get_origin_controller_connection()
        ssh.run([shell], data=setup_sh, capture=False)

    def get_ssh_connection(self) -> SshConnection:
        if False:
            return 10
        'Return an SSH connection for accessing the host.'
        core_ci = self.wait_for_instance()
        settings = SshConnectionDetail(name=core_ci.name, user=core_ci.connection.username, host=core_ci.connection.hostname, port=core_ci.connection.port, identity_file=core_ci.ssh_key.key, python_interpreter=self.python.path)
        if settings.user == 'root':
            become: t.Optional[Become] = None
        elif self.config.become:
            become = SUPPORTED_BECOME_METHODS[self.config.become]()
        else:
            display.warning(f'Defaulting to "sudo" for platform "{self.config.platform}" become support.', unique=True)
            become = Sudo()
        return SshConnection(self.args, settings, become)

    def wait_until_ready(self) -> str:
        if False:
            return 10
        'Wait for instance to respond to SSH, returning the current working directory once connected.'
        core_ci = self.wait_for_instance()
        for dummy in range(1, 90):
            try:
                return self.get_working_directory()
            except SubprocessError as ex:
                display.warning(str(ex))
                time.sleep(10)
        raise HostConnectionError(f'Timeout waiting for {self.config.name} instance {core_ci.instance_id}.')

    def get_controller_target_connections(self) -> list[SshConnection]:
        if False:
            return 10
        'Return SSH connection(s) for accessing the host as a target from the controller.'
        return [self.get_ssh_connection()]

    def get_origin_controller_connection(self) -> SshConnection:
        if False:
            i = 10
            return i + 15
        'Return a connection for accessing the host as a controller from the origin.'
        return self.get_ssh_connection()

    def get_working_directory(self) -> str:
        if False:
            return 10
        'Return the working directory for the host.'
        if not self.pwd:
            ssh = self.get_origin_controller_connection()
            stdout = ssh.run(['pwd'], capture=True)[0]
            if self.args.explain:
                return '/pwd'
            pwd = stdout.strip().splitlines()[-1]
            if not pwd.startswith('/'):
                raise Exception(f'Unexpected current working directory "{pwd}" from "pwd" command output:\n{stdout.strip()}')
            self.pwd = pwd
        return self.pwd

    @property
    def pwd(self) -> t.Optional[str]:
        if False:
            print('Hello World!')
        'Return the cached pwd, if any, otherwise None.'
        return self.cache.get('pwd')

    @pwd.setter
    def pwd(self, value: str) -> None:
        if False:
            i = 10
            return i + 15
        'Cache the given pwd.'
        self.cache['pwd'] = value

class PosixSshProfile(SshTargetHostProfile[PosixSshConfig], PosixProfile[PosixSshConfig]):
    """Host profile for a POSIX SSH instance."""

    def get_controller_target_connections(self) -> list[SshConnection]:
        if False:
            print('Hello World!')
        'Return SSH connection(s) for accessing the host as a target from the controller.'
        settings = SshConnectionDetail(name='target', user=self.config.user, host=self.config.host, port=self.config.port, identity_file=SshKey(self.args).key, python_interpreter=self.python.path)
        return [SshConnection(self.args, settings)]

class WindowsInventoryProfile(SshTargetHostProfile[WindowsInventoryConfig]):
    """Host profile for a Windows inventory."""

    def get_controller_target_connections(self) -> list[SshConnection]:
        if False:
            print('Hello World!')
        'Return SSH connection(s) for accessing the host as a target from the controller.'
        inventory = parse_inventory(self.args, self.config.path)
        hosts = get_hosts(inventory, 'windows')
        identity_file = SshKey(self.args).key
        settings = [SshConnectionDetail(name=name, host=config['ansible_host'], port=22, user=config['ansible_user'], identity_file=identity_file, shell_type='powershell') for (name, config) in hosts.items()]
        if settings:
            details = '\n'.join((f'{ssh.name} {ssh.user}@{ssh.host}:{ssh.port}' for ssh in settings))
            display.info(f'Generated SSH connection details from inventory:\n{details}', verbosity=1)
        return [SshConnection(self.args, setting) for setting in settings]

class WindowsRemoteProfile(RemoteProfile[WindowsRemoteConfig]):
    """Host profile for a Windows remote instance."""

    def wait(self) -> None:
        if False:
            print('Hello World!')
        'Wait for the instance to be ready. Executed before delegation for the controller and after delegation for targets.'
        self.wait_until_ready()

    def get_inventory_variables(self) -> dict[str, t.Optional[t.Union[str, int]]]:
        if False:
            for i in range(10):
                print('nop')
        'Return inventory variables for accessing this host.'
        core_ci = self.wait_for_instance()
        connection = core_ci.connection
        variables: dict[str, t.Optional[t.Union[str, int]]] = dict(ansible_connection='winrm', ansible_pipelining='yes', ansible_winrm_server_cert_validation='ignore', ansible_host=connection.hostname, ansible_port=connection.port, ansible_user=connection.username, ansible_password=connection.password, ansible_ssh_private_key_file=core_ci.ssh_key.key)
        if self.config.version == '2016':
            variables.update(ansible_winrm_transport='ntlm', ansible_winrm_scheme='http', ansible_port='5985')
        return variables

    def wait_until_ready(self) -> None:
        if False:
            return 10
        'Wait for the host to respond to an Ansible module request.'
        core_ci = self.wait_for_instance()
        if not isinstance(self.args, IntegrationConfig):
            return
        inventory = Inventory.create_single_host(sanitize_host_name(self.config.name), self.get_inventory_variables())
        env = ansible_environment(self.args)
        module_name = 'ansible.windows.win_ping'
        with tempfile.NamedTemporaryFile() as inventory_file:
            inventory.write(self.args, inventory_file.name)
            cmd = ['ansible', '-m', module_name, '-i', inventory_file.name, 'all']
            for dummy in range(1, 120):
                try:
                    intercept_python(self.args, self.args.controller_python, cmd, env, capture=True)
                except SubprocessError as ex:
                    display.warning(str(ex))
                    time.sleep(10)
                else:
                    return
        raise HostConnectionError(f'Timeout waiting for {self.config.name} instance {core_ci.instance_id}.')

    def get_controller_target_connections(self) -> list[SshConnection]:
        if False:
            for i in range(10):
                print('nop')
        'Return SSH connection(s) for accessing the host as a target from the controller.'
        core_ci = self.wait_for_instance()
        settings = SshConnectionDetail(name=core_ci.name, host=core_ci.connection.hostname, port=22, user=core_ci.connection.username, identity_file=core_ci.ssh_key.key, shell_type='powershell')
        return [SshConnection(self.args, settings)]

@cache
def get_config_profile_type_map() -> dict[t.Type[HostConfig], t.Type[HostProfile]]:
    if False:
        for i in range(10):
            print('nop')
    'Create and return a mapping of HostConfig types to HostProfile types.'
    return get_type_map(HostProfile, HostConfig)

def create_host_profile(args: EnvironmentConfig, config: HostConfig, controller: bool) -> HostProfile:
    if False:
        return 10
    'Create and return a host profile from the given host configuration.'
    profile_type = get_config_profile_type_map()[type(config)]
    profile = profile_type(args=args, config=config, targets=args.targets if controller else None)
    return profile