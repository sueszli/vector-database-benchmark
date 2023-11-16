"""Test suite used to verify ansible-test is able to run its containers on various container hosts."""
from __future__ import annotations
import abc
import dataclasses
import datetime
import errno
import functools
import json
import os
import pathlib
import pwd
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import time
import typing as t
UNPRIVILEGED_USER_NAME = 'ansible-test'
CGROUP_SYSTEMD = pathlib.Path('/sys/fs/cgroup/systemd')
LOG_PATH = pathlib.Path('/tmp/results')
LOGINUID_NOT_SET = 4294967295
UID = os.getuid()
try:
    LOGINUID = int(pathlib.Path('/proc/self/loginuid').read_text())
    LOGINUID_MISMATCH = LOGINUID != LOGINUID_NOT_SET and LOGINUID != UID
except FileNotFoundError:
    LOGINUID = None
    LOGINUID_MISMATCH = False

def main() -> None:
    if False:
        print('Hello World!')
    'Main program entry point.'
    display.section('Startup check')
    try:
        bootstrap_type = pathlib.Path('/etc/ansible-test.bootstrap').read_text().strip()
    except FileNotFoundError:
        bootstrap_type = 'undefined'
    display.info(f'Bootstrap type: {bootstrap_type}')
    if bootstrap_type != 'remote':
        display.warning('Skipping destructive test on system which is not an ansible-test remote provisioned instance.')
        return
    display.info(f'UID: {UID} / {LOGINUID}')
    if UID != 0:
        raise Exception('This test must be run as root.')
    if not LOGINUID_MISMATCH:
        if LOGINUID is None:
            display.warning('Tests involving loginuid mismatch will be skipped on this host since it does not have audit support.')
        elif LOGINUID == LOGINUID_NOT_SET:
            display.warning('Tests involving loginuid mismatch will be skipped on this host since it is not set.')
        elif LOGINUID == 0:
            raise Exception('Use sudo, su, etc. as a non-root user to become root before running this test.')
        else:
            raise Exception()
    display.section(f'Bootstrapping {os_release}')
    bootstrapper = Bootstrapper.init()
    bootstrapper.run()
    result_dir = LOG_PATH
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir()
    result_dir.chmod(511)
    scenarios = get_test_scenarios()
    results = [run_test(scenario) for scenario in scenarios]
    error_total = 0
    for name in sorted(result_dir.glob('*.log')):
        lines = name.read_text().strip().splitlines()
        error_count = len([line for line in lines if line.startswith('FAIL: ')])
        error_total += error_count
        display.section(f'Log (error_count={error_count!r}/{len(lines)}): {name.name}')
        for line in lines:
            if line.startswith('FAIL: '):
                display.show(line, display.RED)
            else:
                display.show(line)
    error_count = len([result for result in results if result.message])
    error_total += error_count
    duration = datetime.timedelta(seconds=int(sum((result.duration.total_seconds() for result in results))))
    display.section(f'Test Results (error_count={error_count!r}/{len(results)}) [{duration}]')
    for result in results:
        notes = f" <cleanup: {', '.join(result.cleanup)}>" if result.cleanup else ''
        if result.cgroup_dirs:
            notes += f' <cgroup_dirs: {len(result.cgroup_dirs)}>'
        notes += f' [{result.duration}]'
        if result.message:
            display.show(f'FAIL: {result.scenario} {result.message}{notes}', display.RED)
        elif result.duration.total_seconds() >= 90:
            display.show(f'SLOW: {result.scenario}{notes}', display.YELLOW)
        else:
            display.show(f'PASS: {result.scenario}{notes}')
    if error_total:
        sys.exit(1)

def get_test_scenarios() -> list[TestScenario]:
    if False:
        print('Hello World!')
    'Generate and return a list of test scenarios.'
    supported_engines = ('docker', 'podman')
    available_engines = [engine for engine in supported_engines if shutil.which(engine)]
    if not available_engines:
        raise ApplicationError(f"No supported container engines found: {', '.join(supported_engines)}")
    completion_lines = pathlib.Path(os.environ['PYTHONPATH'], '../test/lib/ansible_test/_data/completion/docker.txt').read_text().splitlines()
    entries = {name: value for (name, value) in (parse_completion_entry(line) for line in completion_lines) if name != 'default'}
    unprivileged_user = User.get(UNPRIVILEGED_USER_NAME)
    scenarios: list[TestScenario] = []
    for (container_name, settings) in entries.items():
        image = settings['image']
        cgroup = settings.get('cgroup', 'v1-v2')
        if container_name == 'centos6' and os_release.id == 'alpine':
            continue
        for engine in available_engines:
            disable_selinux = os_release.id == 'fedora' and engine == 'docker' and (cgroup != 'none')
            expose_cgroup_v1 = cgroup == 'v1-only' and get_docker_info(engine).cgroup_version != 1
            debug_systemd = cgroup != 'none'
            probe_cgroups = container_name != 'centos6'
            enable_sha1 = os_release.id == 'rhel' and os_release.version_id.startswith('9.') and (container_name == 'centos6')
            if cgroup != 'none' and get_docker_info(engine).cgroup_version == 1 and (not have_cgroup_systemd()):
                expose_cgroup_v1 = True
            user_scenarios = [UserScenario(ssh=unprivileged_user)]
            if engine == 'podman':
                user_scenarios.append(UserScenario(ssh=ROOT_USER))
                if os_release.id not in ('alpine', 'ubuntu'):
                    user_scenarios.append(UserScenario(remote=unprivileged_user))
                if LOGINUID_MISMATCH:
                    user_scenarios.append(UserScenario())
            for user_scenario in user_scenarios:
                scenarios.append(TestScenario(user_scenario=user_scenario, engine=engine, container_name=container_name, image=image, disable_selinux=disable_selinux, expose_cgroup_v1=expose_cgroup_v1, enable_sha1=enable_sha1, debug_systemd=debug_systemd, probe_cgroups=probe_cgroups))
    return scenarios

def run_test(scenario: TestScenario) -> TestResult:
    if False:
        return 10
    'Run a test scenario and return the test results.'
    display.section(f'Testing {scenario} Started')
    start = time.monotonic()
    integration = ['ansible-test', 'integration', 'split']
    integration_options = ['--target', f'docker:{scenario.container_name}', '--color', '--truncate', '0', '-v']
    target_only_options = []
    if scenario.debug_systemd:
        integration_options.append('--dev-systemd-debug')
    if scenario.probe_cgroups:
        target_only_options = ['--dev-probe-cgroups', str(LOG_PATH)]
    commands = [[*integration, *integration_options, *target_only_options], [*integration, '--controller', 'docker:alpine3', *integration_options]]
    common_env: dict[str, str] = {}
    test_env: dict[str, str] = {}
    if scenario.engine == 'podman':
        if scenario.user_scenario.remote:
            common_env.update(CONTAINER_HOST=f'ssh://{scenario.user_scenario.remote.name}@localhost:22/run/user/{scenario.user_scenario.remote.pwnam.pw_uid}/podman/podman.sock', CONTAINER_SSHKEY=str(pathlib.Path('~/.ssh/id_rsa').expanduser()))
        test_env.update(ANSIBLE_TEST_PREFER_PODMAN='1')
    test_env.update(common_env)
    if scenario.user_scenario.ssh:
        client_become_cmd = ['ssh', f'{scenario.user_scenario.ssh.name}@localhost']
        test_commands = [client_become_cmd + [f'cd ~/ansible; {format_env(test_env)}{sys.executable} bin/{shlex.join(command)}'] for command in commands]
    else:
        client_become_cmd = ['sh', '-c']
        test_commands = [client_become_cmd + [f'{format_env(test_env)}{shlex.join(command)}'] for command in commands]
    prime_storage_command = []
    if scenario.engine == 'podman' and scenario.user_scenario.actual.name == UNPRIVILEGED_USER_NAME:
        actual_become_cmd = ['ssh', f'{scenario.user_scenario.actual.name}@localhost']
        prime_storage_command = actual_become_cmd + prepare_prime_podman_storage()
    message = ''
    if scenario.expose_cgroup_v1:
        prepare_cgroup_systemd(scenario.user_scenario.actual.name, scenario.engine)
    try:
        if prime_storage_command:
            retry_command(lambda : run_command(*prime_storage_command), retry_any_error=True)
        if scenario.disable_selinux:
            run_command('setenforce', 'permissive')
        if scenario.enable_sha1:
            run_command('update-crypto-policies', '--set', 'DEFAULT:SHA1')
        for test_command in test_commands:
            retry_command(lambda : run_command(*test_command))
    except SubprocessError as ex:
        message = str(ex)
        display.error(f'{scenario} {message}')
    finally:
        if scenario.enable_sha1:
            run_command('update-crypto-policies', '--set', 'DEFAULT')
        if scenario.disable_selinux:
            run_command('setenforce', 'enforcing')
        if scenario.expose_cgroup_v1:
            dirs = remove_cgroup_systemd()
        else:
            dirs = list_group_systemd()
        cleanup_command = [scenario.engine, 'rmi', '-f', scenario.image]
        try:
            retry_command(lambda : run_command(*client_become_cmd + [f'{format_env(common_env)}{shlex.join(cleanup_command)}']), retry_any_error=True)
        except SubprocessError as ex:
            display.error(str(ex))
        cleanup = cleanup_podman() if scenario.engine == 'podman' else tuple()
    finish = time.monotonic()
    duration = datetime.timedelta(seconds=int(finish - start))
    display.section(f'Testing {scenario} Completed in {duration}')
    return TestResult(scenario=scenario, message=message, cleanup=cleanup, duration=duration, cgroup_dirs=tuple((str(path) for path in dirs)))

def prepare_prime_podman_storage() -> list[str]:
    if False:
        i = 10
        return i + 15
    'Partially prime podman storage and return a command to complete the remainder.'
    prime_storage_command = ['rm -rf ~/.local/share/containers; STORAGE_DRIVER=overlay podman pull quay.io/bedrock/alpine:3.16.2']
    test_containers = pathlib.Path(f'~{UNPRIVILEGED_USER_NAME}/.local/share/containers').expanduser()
    if test_containers.is_dir():
        rmtree(test_containers)
    return prime_storage_command

def cleanup_podman() -> tuple[str, ...]:
    if False:
        while True:
            i = 10
    'Cleanup podman processes and files on disk.'
    cleanup = []
    for remaining in range(3, -1, -1):
        processes = [(int(item[0]), item[1]) for item in [item.split(maxsplit=1) for item in run_command('ps', '-A', '-o', 'pid,comm', capture=True).stdout.splitlines()] if pathlib.Path(item[1].split()[0]).name in ('catatonit', 'podman', 'conmon')]
        if not processes:
            break
        for (pid, name) in processes:
            display.info(f'Killing "{name}" ({pid}) ...')
            try:
                os.kill(pid, signal.SIGTERM if remaining > 1 else signal.SIGKILL)
            except ProcessLookupError:
                pass
            cleanup.append(name)
        time.sleep(1)
    else:
        raise Exception('failed to kill all matching processes')
    uid = pwd.getpwnam(UNPRIVILEGED_USER_NAME).pw_uid
    container_tmp = pathlib.Path(f'/tmp/containers-user-{uid}')
    podman_tmp = pathlib.Path(f'/tmp/podman-run-{uid}')
    user_config = pathlib.Path(f'~{UNPRIVILEGED_USER_NAME}/.config').expanduser()
    user_local = pathlib.Path(f'~{UNPRIVILEGED_USER_NAME}/.local').expanduser()
    if container_tmp.is_dir():
        rmtree(container_tmp)
    if podman_tmp.is_dir():
        rmtree(podman_tmp)
    if user_config.is_dir():
        rmtree(user_config)
    if user_local.is_dir():
        rmtree(user_local)
    return tuple(sorted(set(cleanup)))

def have_cgroup_systemd() -> bool:
    if False:
        i = 10
        return i + 15
    'Return True if the container host has a systemd cgroup.'
    return pathlib.Path(CGROUP_SYSTEMD).is_dir()

def prepare_cgroup_systemd(username: str, engine: str) -> None:
    if False:
        while True:
            i = 10
    'Prepare the systemd cgroup.'
    CGROUP_SYSTEMD.mkdir()
    run_command('mount', 'cgroup', '-t', 'cgroup', str(CGROUP_SYSTEMD), '-o', 'none,name=systemd,xattr', capture=True)
    if engine == 'podman':
        run_command('chown', '-R', f'{username}:{username}', str(CGROUP_SYSTEMD))
    run_command('find', str(CGROUP_SYSTEMD), '-type', 'd', '-exec', 'ls', '-l', '{}', ';')

def list_group_systemd() -> list[pathlib.Path]:
    if False:
        while True:
            i = 10
    'List the systemd cgroup.'
    dirs = set()
    for (dirpath, dirnames, filenames) in os.walk(CGROUP_SYSTEMD, topdown=False):
        for dirname in dirnames:
            target_path = pathlib.Path(dirpath, dirname)
            display.info(f'dir: {target_path}')
            dirs.add(target_path)
    return sorted(dirs)

def remove_cgroup_systemd() -> list[pathlib.Path]:
    if False:
        i = 10
        return i + 15
    'Remove the systemd cgroup.'
    dirs = set()
    for sleep_seconds in range(1, 10):
        try:
            for (dirpath, dirnames, filenames) in os.walk(CGROUP_SYSTEMD, topdown=False):
                for dirname in dirnames:
                    target_path = pathlib.Path(dirpath, dirname)
                    display.info(f'rmdir: {target_path}')
                    dirs.add(target_path)
                    target_path.rmdir()
        except OSError as ex:
            if ex.errno != errno.EBUSY:
                raise
            error = str(ex)
        else:
            break
        display.warning(f'{error} -- sleeping for {sleep_seconds} second(s) before trying again ...')
        time.sleep(sleep_seconds)
    time.sleep(1)
    run_command('umount', str(CGROUP_SYSTEMD))
    CGROUP_SYSTEMD.rmdir()
    time.sleep(1)
    cgroup = pathlib.Path('/proc/self/cgroup').read_text()
    if 'systemd' in cgroup:
        raise Exception('systemd hierarchy detected')
    return sorted(dirs)

def rmtree(path: pathlib.Path) -> None:
    if False:
        i = 10
        return i + 15
    'Wrapper around shutil.rmtree with additional error handling.'
    for retries in range(10, -1, -1):
        try:
            display.info(f'rmtree: {path} ({retries} attempts remaining) ... ')
            shutil.rmtree(path)
        except Exception:
            if not path.exists():
                display.info(f'rmtree: {path} (not found)')
                return
            if not path.is_dir():
                display.info(f'rmtree: {path} (not a directory)')
                return
            if retries:
                continue
            raise
        else:
            display.info(f'rmtree: {path} (done)')
            return

def format_env(env: dict[str, str]) -> str:
    if False:
        i = 10
        return i + 15
    'Format an env dict for injection into a shell command and return the resulting string.'
    if env:
        return ' '.join((f'{shlex.quote(key)}={shlex.quote(value)}' for (key, value) in env.items())) + ' '
    return ''

class DockerInfo:
    """The results of `docker info` for the container runtime."""

    def __init__(self, data: dict[str, t.Any]) -> None:
        if False:
            i = 10
            return i + 15
        self.data = data

    @property
    def cgroup_version(self) -> int:
        if False:
            while True:
                i = 10
        'The cgroup version of the container host.'
        data = self.data
        host = data.get('host')
        if host:
            version = int(host['cgroupVersion'].lstrip('v'))
        else:
            version = int(data['CgroupVersion'])
        return version

@functools.lru_cache
def get_docker_info(engine: str) -> DockerInfo:
    if False:
        return 10
    'Return info for the current container runtime. The results are cached.'
    return DockerInfo(json.loads(run_command(engine, 'info', '--format', '{{ json . }}', capture=True).stdout))

@dataclasses.dataclass(frozen=True)
class User:
    name: str
    pwnam: pwd.struct_passwd

    @classmethod
    def get(cls, name: str) -> User:
        if False:
            while True:
                i = 10
        return User(name=name, pwnam=pwd.getpwnam(name))

@dataclasses.dataclass(frozen=True)
class UserScenario:
    ssh: User = None
    remote: User = None

    @property
    def actual(self) -> User:
        if False:
            while True:
                i = 10
        return self.remote or self.ssh or ROOT_USER

@dataclasses.dataclass(frozen=True)
class TestScenario:
    user_scenario: UserScenario
    engine: str
    container_name: str
    image: str
    disable_selinux: bool
    expose_cgroup_v1: bool
    enable_sha1: bool
    debug_systemd: bool
    probe_cgroups: bool

    @property
    def tags(self) -> tuple[str, ...]:
        if False:
            for i in range(10):
                print('nop')
        tags = []
        if self.user_scenario.ssh:
            tags.append(f'ssh: {self.user_scenario.ssh.name}')
        if self.user_scenario.remote:
            tags.append(f'remote: {self.user_scenario.remote.name}')
        if self.disable_selinux:
            tags.append('selinux: permissive')
        if self.expose_cgroup_v1:
            tags.append('cgroup: v1')
        if self.enable_sha1:
            tags.append('sha1: enabled')
        return tuple(tags)

    @property
    def tag_label(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return ' '.join((f'[{tag}]' for tag in self.tags))

    def __str__(self):
        if False:
            print('Hello World!')
        return f'[{self.container_name}] ({self.engine}) {self.tag_label}'.strip()

@dataclasses.dataclass(frozen=True)
class TestResult:
    scenario: TestScenario
    message: str
    cleanup: tuple[str, ...]
    duration: datetime.timedelta
    cgroup_dirs: tuple[str, ...]

def parse_completion_entry(value: str) -> tuple[str, dict[str, str]]:
    if False:
        print('Hello World!')
    'Parse the given completion entry, returning the entry name and a dictionary of key/value settings.'
    values = value.split()
    name = values[0]
    data = {kvp[0]: kvp[1] if len(kvp) > 1 else '' for kvp in [item.split('=', 1) for item in values[1:]]}
    return (name, data)

@dataclasses.dataclass(frozen=True)
class SubprocessResult:
    """Result from execution of a subprocess."""
    command: list[str]
    stdout: str
    stderr: str
    status: int

class ApplicationError(Exception):
    """An application error."""

    def __init__(self, message: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.message = message
        super().__init__(message)

class SubprocessError(ApplicationError):
    """An error from executing a subprocess."""

    def __init__(self, result: SubprocessResult) -> None:
        if False:
            i = 10
            return i + 15
        self.result = result
        message = f'Command `{shlex.join(result.command)}` exited with status: {result.status}'
        stdout = (result.stdout or '').strip()
        stderr = (result.stderr or '').strip()
        if stdout:
            message += f'\n>>> Standard Output\n{stdout}'
        if stderr:
            message += f'\n>>> Standard Error\n{stderr}'
        super().__init__(message)

class ProgramNotFoundError(ApplicationError):
    """A required program was not found."""

    def __init__(self, name: str) -> None:
        if False:
            i = 10
            return i + 15
        self.name = name
        super().__init__(f'Missing program: {name}')

class Display:
    """Display interface for sending output to the console."""
    CLEAR = '\x1b[0m'
    RED = '\x1b[31m'
    GREEN = '\x1b[32m'
    YELLOW = '\x1b[33m'
    BLUE = '\x1b[34m'
    PURPLE = '\x1b[35m'
    CYAN = '\x1b[36m'

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.sensitive: set[str] = set()

    def section(self, message: str) -> None:
        if False:
            while True:
                i = 10
        'Print a section message to the console.'
        self.show(f'==> {message}', color=self.BLUE)

    def subsection(self, message: str) -> None:
        if False:
            print('Hello World!')
        'Print a subsection message to the console.'
        self.show(f'--> {message}', color=self.CYAN)

    def fatal(self, message: str) -> None:
        if False:
            print('Hello World!')
        'Print a fatal message to the console.'
        self.show(f'FATAL: {message}', color=self.RED)

    def error(self, message: str) -> None:
        if False:
            i = 10
            return i + 15
        'Print an error message to the console.'
        self.show(f'ERROR: {message}', color=self.RED)

    def warning(self, message: str) -> None:
        if False:
            return 10
        'Print a warning message to the console.'
        self.show(f'WARNING: {message}', color=self.PURPLE)

    def info(self, message: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Print an info message to the console.'
        self.show(f'INFO: {message}', color=self.YELLOW)

    def show(self, message: str, color: str | None=None) -> None:
        if False:
            return 10
        'Print a message to the console.'
        for item in self.sensitive:
            message = message.replace(item, '*' * len(item))
        print(f'{color or self.CLEAR}{message}{self.CLEAR}', flush=True)

def run_module(module: str, args: dict[str, t.Any]) -> SubprocessResult:
    if False:
        return 10
    'Run the specified Ansible module and return the result.'
    return run_command('ansible', '-m', module, '-v', '-a', json.dumps(args), 'localhost')

def retry_command(func: t.Callable[[], SubprocessResult], attempts: int=3, retry_any_error: bool=False) -> SubprocessResult:
    if False:
        while True:
            i = 10
    'Run the given command function up to the specified number of attempts when the failure is due to an SSH error.'
    for attempts_remaining in range(attempts - 1, -1, -1):
        try:
            return func()
        except SubprocessError as ex:
            if ex.result.command[0] == 'ssh' and ex.result.status == 255 and attempts_remaining:
                display.warning('Command failed due to an SSH error. Waiting a few seconds before retrying.')
                time.sleep(3)
                continue
            if retry_any_error:
                display.warning('Command failed. Waiting a few seconds before retrying.')
                time.sleep(3)
                continue
            raise

def run_command(*command: str, data: str | None=None, stdin: int | t.IO[bytes] | None=None, env: dict[str, str] | None=None, capture: bool=False) -> SubprocessResult:
    if False:
        print('Hello World!')
    'Run the specified command and return the result.'
    stdin = subprocess.PIPE if data else stdin or subprocess.DEVNULL
    stdout = subprocess.PIPE if capture else None
    stderr = subprocess.PIPE if capture else None
    display.subsection(f'Run command: {shlex.join(command)}')
    try:
        with subprocess.Popen(args=command, stdin=stdin, stdout=stdout, stderr=stderr, env=env, text=True) as process:
            (process_stdout, process_stderr) = process.communicate(data)
            process_status = process.returncode
    except FileNotFoundError:
        raise ProgramNotFoundError(command[0]) from None
    result = SubprocessResult(command=list(command), stdout=process_stdout, stderr=process_stderr, status=process_status)
    if process.returncode != 0:
        raise SubprocessError(result)
    return result

class Bootstrapper(metaclass=abc.ABCMeta):
    """Bootstrapper for remote instances."""

    @classmethod
    def install_podman(cls) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Return True if podman will be installed.'
        return False

    @classmethod
    def install_docker(cls) -> bool:
        if False:
            print('Hello World!')
        'Return True if docker will be installed.'
        return False

    @classmethod
    def usable(cls) -> bool:
        if False:
            i = 10
            return i + 15
        'Return True if the bootstrapper can be used, otherwise False.'
        return False

    @classmethod
    def init(cls) -> t.Type[Bootstrapper]:
        if False:
            return 10
        'Return a bootstrapper type appropriate for the current system.'
        for bootstrapper in cls.__subclasses__():
            if bootstrapper.usable():
                return bootstrapper
        display.warning('No supported bootstrapper found.')
        return Bootstrapper

    @classmethod
    def run(cls) -> None:
        if False:
            while True:
                i = 10
        'Run the bootstrapper.'
        cls.configure_root_user()
        cls.configure_unprivileged_user()
        cls.configure_source_trees()
        cls.configure_ssh_keys()
        cls.configure_podman_remote()

    @classmethod
    def configure_root_user(cls) -> None:
        if False:
            i = 10
            return i + 15
        'Configure the root user to run tests.'
        root_password_status = run_command('passwd', '--status', 'root', capture=True)
        root_password_set = root_password_status.stdout.split()[1]
        if root_password_set not in ('P', 'PS'):
            root_password = run_command('openssl', 'passwd', '-5', '-stdin', data=secrets.token_hex(8), capture=True).stdout.strip()
            run_module('user', dict(user='root', password=root_password))

    @classmethod
    def configure_unprivileged_user(cls) -> None:
        if False:
            print('Hello World!')
        'Configure the unprivileged user to run tests.'
        unprivileged_password = run_command('openssl', 'passwd', '-5', '-stdin', data=secrets.token_hex(8), capture=True).stdout.strip()
        run_module('user', dict(user=UNPRIVILEGED_USER_NAME, password=unprivileged_password, groups=['docker'] if cls.install_docker() else [], append=True))
        if os_release.id == 'alpine':
            start = 165535
            end = start + 65535
            id_range = f'{start}-{end}'
            run_command('usermod', '--add-subuids', id_range, '--add-subgids', id_range, UNPRIVILEGED_USER_NAME)

    @classmethod
    def configure_source_trees(cls):
        if False:
            for i in range(10):
                print('nop')
        'Configure the source trees needed to run tests for both root and the unprivileged user.'
        current_ansible = pathlib.Path(os.environ['PYTHONPATH']).parent
        root_ansible = pathlib.Path('~').expanduser() / 'ansible'
        test_ansible = pathlib.Path(f'~{UNPRIVILEGED_USER_NAME}').expanduser() / 'ansible'
        if current_ansible != root_ansible:
            display.info(f'copying {current_ansible} -> {root_ansible} ...')
            rmtree(root_ansible)
            shutil.copytree(current_ansible, root_ansible)
            run_command('chown', '-R', 'root:root', str(root_ansible))
        display.info(f'copying {current_ansible} -> {test_ansible} ...')
        rmtree(test_ansible)
        shutil.copytree(current_ansible, test_ansible)
        run_command('chown', '-R', f'{UNPRIVILEGED_USER_NAME}:{UNPRIVILEGED_USER_NAME}', str(test_ansible))
        paths = [pathlib.Path(test_ansible)]
        for (root, dir_names, file_names) in os.walk(test_ansible):
            paths.extend((pathlib.Path(root, dir_name) for dir_name in dir_names))
            paths.extend((pathlib.Path(root, file_name) for file_name in file_names))
        user = pwd.getpwnam(UNPRIVILEGED_USER_NAME)
        uid = user.pw_uid
        gid = user.pw_gid
        for path in paths:
            os.chown(path, uid, gid)

    @classmethod
    def configure_ssh_keys(cls) -> None:
        if False:
            i = 10
            return i + 15
        'Configure SSH keys needed to run tests.'
        user = pwd.getpwnam(UNPRIVILEGED_USER_NAME)
        uid = user.pw_uid
        gid = user.pw_gid
        current_rsa_pub = pathlib.Path('~/.ssh/id_rsa.pub').expanduser()
        test_authorized_keys = pathlib.Path(f'~{UNPRIVILEGED_USER_NAME}/.ssh/authorized_keys').expanduser()
        test_authorized_keys.parent.mkdir(mode=493, parents=True, exist_ok=True)
        os.chown(test_authorized_keys.parent, uid, gid)
        shutil.copyfile(current_rsa_pub, test_authorized_keys)
        os.chown(test_authorized_keys, uid, gid)
        test_authorized_keys.chmod(mode=420)

    @classmethod
    def configure_podman_remote(cls) -> None:
        if False:
            while True:
                i = 10
        'Configure podman remote support.'
        if os_release.id in ('alpine', 'ubuntu'):
            return
        retry_command(lambda : run_command('ssh', f'{UNPRIVILEGED_USER_NAME}@localhost', 'systemctl', '--user', 'enable', '--now', 'podman.socket'))
        run_command('loginctl', 'enable-linger', UNPRIVILEGED_USER_NAME)

class DnfBootstrapper(Bootstrapper):
    """Bootstrapper for dnf based systems."""

    @classmethod
    def install_podman(cls) -> bool:
        if False:
            return 10
        'Return True if podman will be installed.'
        return True

    @classmethod
    def install_docker(cls) -> bool:
        if False:
            i = 10
            return i + 15
        'Return True if docker will be installed.'
        return os_release.id != 'rhel'

    @classmethod
    def usable(cls) -> bool:
        if False:
            i = 10
            return i + 15
        'Return True if the bootstrapper can be used, otherwise False.'
        return bool(shutil.which('dnf'))

    @classmethod
    def run(cls) -> None:
        if False:
            while True:
                i = 10
        'Run the bootstrapper.'
        packages = ['podman', 'crun']
        if cls.install_docker():
            packages.append('moby-engine')
        if os_release.id == 'fedora' and os_release.version_id == '36':
            packages.append('netavark-1.0.2')
        if os_release.id == 'rhel':
            run_command('dnf', 'update', '-y', 'policycoreutils')
        run_command('dnf', 'install', '-y', *packages)
        if cls.install_docker():
            run_command('systemctl', 'start', 'docker')
        if os_release.id == 'rhel' and os_release.version_id.startswith('8.'):
            conf = pathlib.Path('/usr/share/containers/containers.conf').read_text()
            conf = re.sub('^runtime .*', 'runtime = "crun"', conf, flags=re.MULTILINE)
            pathlib.Path('/etc/containers/containers.conf').write_text(conf)
        super().run()

class AptBootstrapper(Bootstrapper):
    """Bootstrapper for apt based systems."""

    @classmethod
    def install_podman(cls) -> bool:
        if False:
            print('Hello World!')
        'Return True if podman will be installed.'
        return not (os_release.id == 'ubuntu' and os_release.version_id == '20.04')

    @classmethod
    def install_docker(cls) -> bool:
        if False:
            while True:
                i = 10
        'Return True if docker will be installed.'
        return True

    @classmethod
    def usable(cls) -> bool:
        if False:
            return 10
        'Return True if the bootstrapper can be used, otherwise False.'
        return bool(shutil.which('apt-get'))

    @classmethod
    def run(cls) -> None:
        if False:
            return 10
        'Run the bootstrapper.'
        apt_env = os.environ.copy()
        apt_env.update(DEBIAN_FRONTEND='noninteractive')
        packages = ['docker.io']
        if cls.install_podman():
            packages.extend(('podman', 'crun', 'uidmap', 'slirp4netns'))
        run_command('apt-get', 'install', *packages, '-y', '--no-install-recommends', env=apt_env)
        super().run()

class ApkBootstrapper(Bootstrapper):
    """Bootstrapper for apk based systems."""

    @classmethod
    def install_podman(cls) -> bool:
        if False:
            while True:
                i = 10
        'Return True if podman will be installed.'
        return True

    @classmethod
    def install_docker(cls) -> bool:
        if False:
            return 10
        'Return True if docker will be installed.'
        return True

    @classmethod
    def usable(cls) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Return True if the bootstrapper can be used, otherwise False.'
        return bool(shutil.which('apk'))

    @classmethod
    def run(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Run the bootstrapper.'
        packages = ['docker', 'podman', 'openssl', 'crun', 'ip6tables']
        run_command('apk', 'add', *packages)
        run_command('apk', 'upgrade', '-U', '--repository=http://dl-cdn.alpinelinux.org/alpine/edge/community', 'crun')
        run_command('service', 'docker', 'start')
        run_command('modprobe', 'tun')
        super().run()

@dataclasses.dataclass(frozen=True)
class OsRelease:
    """Operating system identification."""
    id: str
    version_id: str

    @staticmethod
    def init() -> OsRelease:
        if False:
            print('Hello World!')
        'Detect the current OS release and return the result.'
        lines = run_command('sh', '-c', '. /etc/os-release && echo $ID && echo $VERSION_ID', capture=True).stdout.splitlines()
        result = OsRelease(id=lines[0], version_id=lines[1])
        display.show(f'Detected OS "{result.id}" version "{result.version_id}".')
        return result
display = Display()
os_release = OsRelease.init()
ROOT_USER = User.get('root')
if __name__ == '__main__':
    main()