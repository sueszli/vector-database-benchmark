from __future__ import annotations
import contextlib
import http
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Callable, Generator, Literal, NamedTuple, overload
import click
import requests
if TYPE_CHECKING:
    import docker
DARWIN = sys.platform == 'darwin'
APPLE_ARM64 = DARWIN and platform.processor() in {'arm', 'arm64'}
USE_COLIMA = bool(shutil.which('colima'))
if USE_COLIMA:
    RAW_SOCKET_PATH = os.path.expanduser('~/.colima/default/docker.sock')
else:
    RAW_SOCKET_PATH = '/var/run/docker.sock'

@contextlib.contextmanager
def get_docker_client() -> Generator[docker.DockerClient, None, None]:
    if False:
        return 10
    import docker
    with contextlib.closing(docker.DockerClient(base_url=f'unix://{RAW_SOCKET_PATH}')) as client:
        try:
            client.ping()
        except (requests.exceptions.ConnectionError, docker.errors.APIError):
            if DARWIN:
                if USE_COLIMA:
                    click.echo('Attempting to start colima...')
                    subprocess.check_call(('python3', '-uS', f'{os.path.dirname(__file__)}/../../../../scripts/start-colima.py'))
                else:
                    click.echo('Attempting to start docker...')
                    subprocess.check_call(('open', '-a', '/Applications/Docker.app', '--args', '--unattended'))
            else:
                raise click.ClickException('Make sure docker is running.')
            max_wait = 60
            timeout = time.monotonic() + max_wait
            click.echo(f'Waiting for docker to be ready.... (timeout in {max_wait}s)')
            while time.monotonic() < timeout:
                time.sleep(1)
                try:
                    client.ping()
                except (requests.exceptions.ConnectionError, docker.errors.APIError):
                    continue
                else:
                    break
            else:
                raise click.ClickException('Failed to start docker.')
        yield client

@overload
def get_or_create(client: docker.DockerClient, thing: Literal['network'], name: str) -> docker.models.networks.Network:
    if False:
        i = 10
        return i + 15
    ...

@overload
def get_or_create(client: docker.DockerClient, thing: Literal['volume'], name: str) -> docker.models.volumes.Volume:
    if False:
        print('Hello World!')
    ...

def get_or_create(client: docker.DockerClient, thing: Literal['network', 'volume'], name: str) -> docker.models.networks.Network | docker.models.volumes.Volume:
    if False:
        i = 10
        return i + 15
    from docker.errors import NotFound
    try:
        return getattr(client, thing + 's').get(name)
    except NotFound:
        click.secho(f"> Creating '{name}' {thing}", err=True, fg='yellow')
        return getattr(client, thing + 's').create(name)

def retryable_pull(client: docker.DockerClient, image: str, max_attempts: int=5) -> None:
    if False:
        return 10
    from docker.errors import APIError
    current_attempt = 0
    while True:
        try:
            client.images.pull(image)
        except APIError:
            if current_attempt + 1 >= max_attempts:
                raise
            current_attempt = current_attempt + 1
            continue
        else:
            break

def ensure_interface(ports: dict[str, int | tuple[str, int]]) -> dict[str, tuple[str, int]]:
    if False:
        i = 10
        return i + 15
    rv = {}
    for (k, v) in ports.items():
        if not isinstance(v, tuple):
            v = ('127.0.0.1', v)
        rv[k] = v
    return rv

@click.group()
def devservices() -> None:
    if False:
        i = 10
        return i + 15
    '\n    Manage dependent development services required for Sentry.\n\n    Do not use in production!\n    '
    os.environ['SENTRY_SKIP_BACKEND_VALIDATION'] = '1'

@devservices.command()
@click.option('--project', default='sentry')
@click.argument('service', nargs=1)
def attach(project: str, service: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Run a single devservice in the foreground.\n\n    Accepts a single argument, the name of the service to spawn. The service\n    will run with output printed to your terminal, and the ability to kill it\n    with ^C. This is used in devserver.\n\n    Note: This does not update images, you will have to use `devservices up`\n    for that.\n    '
    from sentry.runner import configure
    configure()
    containers = _prepare_containers(project, silent=True)
    if service not in containers:
        raise click.ClickException(f'Service `{service}` is not known or not enabled.')
    with get_docker_client() as docker_client:
        container = _start_service(docker_client, service, containers, project, always_start=True)
        if container is None:
            raise click.ClickException(f'No containers found for service `{service}`.')

        def exit_handler(*_: Any) -> None:
            if False:
                return 10
            try:
                click.echo(f'Stopping {service}')
                container.stop()
                click.echo(f'Removing {service}')
                container.remove()
            except KeyboardInterrupt:
                pass
        signal.signal(signal.SIGINT, exit_handler)
        signal.signal(signal.SIGTERM, exit_handler)
        for line in container.logs(stream=True, since=int(time.time() - 20)):
            click.echo(line, nl=False)

@devservices.command()
@click.argument('services', nargs=-1)
@click.option('--project', default='sentry')
@click.option('--exclude', multiple=True, help='Service to ignore and not run. Repeatable option.')
@click.option('--skip-only-if', is_flag=True, default=False, help="Skip 'only_if' checks for services")
@click.option('--recreate', is_flag=True, default=False, help='Recreate containers that are already running.')
def up(services: list[str], project: str, exclude: list[str], skip_only_if: bool, recreate: bool) -> None:
    if False:
        while True:
            i = 10
    '\n    Run/update all devservices in the background.\n\n    The default is everything, however you may pass positional arguments to specify\n    an explicit list of services to bring up.\n\n    You may also exclude services, for example: --exclude redis --exclude postgres.\n    '
    from sentry.runner import configure
    configure()
    containers = _prepare_containers(project, skip_only_if=skip_only_if or len(services) > 0, silent=True)
    selected_services = set()
    if services:
        for service in services:
            if service not in containers:
                click.secho(f'Service `{service}` is not known or not enabled.\n', err=True, fg='red')
                click.secho('Services that are available:\n' + '\n'.join(containers.keys()) + '\n', err=True)
                raise click.Abort()
            selected_services.add(service)
    else:
        selected_services = set(containers.keys())
    for service in exclude:
        if service not in containers:
            click.secho(f'Service `{service}` is not known or not enabled.\n', err=True, fg='red')
            click.secho('Services that are available:\n' + '\n'.join(containers.keys()) + '\n', err=True)
            raise click.Abort()
        selected_services.remove(service)
    with get_docker_client() as docker_client:
        get_or_create(docker_client, 'network', project)
        with ThreadPoolExecutor(max_workers=len(selected_services)) as executor:
            futures = []
            for name in selected_services:
                futures.append(executor.submit(_start_service, docker_client, name, containers, project, False, recreate))
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    click.secho(f'> Failed to start service: {e}', err=True, fg='red')
                    raise
    with ThreadPoolExecutor(max_workers=len(selected_services)) as executor:
        futures = []
        for name in selected_services:
            futures.append(executor.submit(check_health, name, containers[name]))
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                click.secho(f'> Failed to check health: {e}', err=True, fg='red')
                raise

def _prepare_containers(project: str, skip_only_if: bool=False, silent: bool=False) -> dict[str, Any]:
    if False:
        for i in range(10):
            print('nop')
    from django.conf import settings
    from sentry import options as sentry_options
    containers = {}
    for (name, option_builder) in settings.SENTRY_DEVSERVICES.items():
        options = option_builder(settings, sentry_options)
        only_if = options.pop('only_if', True)
        if not skip_only_if and (not only_if):
            if not silent:
                click.secho(f'! Skipping {name} due to only_if condition', err=True, fg='cyan')
            continue
        options['network'] = project
        options['detach'] = True
        options['name'] = project + '_' + name
        options.setdefault('ports', {})
        options.setdefault('environment', {})
        options.setdefault('restart_policy', {'Name': 'unless-stopped'})
        options['ports'] = ensure_interface(options['ports'])
        options['extra_hosts'] = {'host.docker.internal': 'host-gateway'}
        containers[name] = options
    return containers

@overload
def _start_service(client: docker.DockerClient, name: str, containers: dict[str, Any], project: str, always_start: Literal[False]=..., recreate: bool=False) -> docker.models.containers.Container:
    if False:
        print('Hello World!')
    ...

@overload
def _start_service(client: docker.DockerClient, name: str, containers: dict[str, Any], project: str, always_start: bool=False, recreate: bool=False) -> docker.models.containers.Container | None:
    if False:
        i = 10
        return i + 15
    ...

def _start_service(client: docker.DockerClient, name: str, containers: dict[str, Any], project: str, always_start: bool=False, recreate: bool=False) -> docker.models.containers.Container | None:
    if False:
        while True:
            i = 10
    from docker.errors import NotFound
    options = containers[name]
    with_devserver = options.pop('with_devserver', False)
    if with_devserver and (not always_start):
        click.secho(f"> Not starting container '{options['name']}' because it should be started on-demand with devserver.", fg='yellow')
        return None
    container = None
    try:
        container = client.containers.get(options['name'])
    except NotFound:
        pass
    if container is not None:
        if not recreate and container.status == 'running':
            click.secho(f"> Container '{options['name']}' is already running", fg='yellow')
            return container
        click.secho(f"> Stopping container '{container.name}'", fg='yellow')
        container.stop()
        click.secho(f"> Removing container '{container.name}'", fg='yellow')
        container.remove()
    for (key, value) in list(options['environment'].items()):
        options['environment'][key] = value.format(containers=containers)
    click.secho(f"> Pulling image '{options['image']}'", fg='green')
    retryable_pull(client, options['image'])
    for mount in list(options.get('volumes', {}).keys()):
        if '/' not in mount:
            get_or_create(client, 'volume', project + '_' + mount)
            options['volumes'][project + '_' + mount] = options['volumes'].pop(mount)
    listening = ''
    if options['ports']:
        listening = '(listening: %s)' % ', '.join(map(str, options['ports'].values()))
    click.secho(f"> Creating container '{options['name']}'", fg='yellow')
    container = client.containers.create(**options)
    click.secho(f"> Starting container '{container.name}' {listening}", fg='yellow')
    container.start()
    return container

@devservices.command()
@click.option('--project', default='sentry')
@click.argument('service', nargs=-1)
def down(project: str, service: list[str]) -> None:
    if False:
        while True:
            i = 10
    '\n    Shut down services without deleting their underlying data.\n    Useful if you want to temporarily relieve resources on your computer.\n\n    The default is everything, however you may pass positional arguments to specify\n    an explicit list of services to bring down.\n    '

    def _down(container: docker.models.containers.Container) -> None:
        if False:
            while True:
                i = 10
        click.secho(f"> Stopping '{container.name}' container", fg='red')
        container.stop()
        click.secho(f"> Removing '{container.name}' container", fg='red')
        container.remove()
    containers = []
    prefix = f'{project}_'
    with get_docker_client() as docker_client:
        for container in docker_client.containers.list(all=True):
            if not container.name.startswith(prefix):
                continue
            if service and (not container.name[len(prefix):] in service):
                continue
            containers.append(container)
        with ThreadPoolExecutor(max_workers=len(containers) or 1) as executor:
            futures = []
            for container in containers:
                futures.append(executor.submit(_down, container))
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    click.secho(f'> Failed to stop service: {e}', err=True, fg='red')
                    raise

@devservices.command()
@click.option('--project', default='sentry')
@click.argument('services', nargs=-1)
def rm(project: str, services: list[str]) -> None:
    if False:
        for i in range(10):
            print('nop')
    "\n    Shut down and delete all services and associated data.\n    Useful if you'd like to start with a fresh slate.\n\n    The default is everything, however you may pass positional arguments to specify\n    an explicit list of services to remove.\n    "
    from docker.errors import NotFound
    from sentry.runner import configure
    configure()
    containers = _prepare_containers(project, skip_only_if=len(services) > 0, silent=True)
    if services:
        selected_containers = {}
        for service in services:
            if service not in containers:
                click.secho(f'Service `{service}` is not known or not enabled.\n', err=True, fg='red')
                click.secho('Services that are available:\n' + '\n'.join(containers.keys()) + '\n', err=True)
                raise click.Abort()
            selected_containers[service] = containers[service]
        containers = selected_containers
    click.confirm('\nThis will delete these services and all of their data:\n\n%s\n\nAre you sure you want to continue?' % '\n'.join(containers.keys()), abort=True)
    with get_docker_client() as docker_client:
        volume_to_service = {}
        for (service_name, container_options) in containers.items():
            try:
                container = docker_client.containers.get(container_options['name'])
            except NotFound:
                click.secho("> WARNING: non-existent container '%s'" % container_options['name'], err=True, fg='yellow')
                continue
            click.secho("> Stopping '%s' container" % container_options['name'], err=True, fg='red')
            container.stop()
            click.secho("> Removing '%s' container" % container_options['name'], err=True, fg='red')
            container.remove()
            for volume in container_options.get('volumes') or ():
                volume_to_service[volume] = service_name
        prefix = project + '_'
        for volume in docker_client.volumes.list():
            if volume.name.startswith(prefix):
                local_name = volume.name[len(prefix):]
                if not services or volume_to_service.get(local_name) in services:
                    click.secho("> Removing '%s' volume" % volume.name, err=True, fg='red')
                    volume.remove()
        if not services:
            try:
                network = docker_client.networks.get(project)
            except NotFound:
                pass
            else:
                click.secho("> Removing '%s' network" % network.name, err=True, fg='red')
                network.remove()

def check_health(service_name: str, options: dict[str, Any]) -> None:
    if False:
        while True:
            i = 10
    healthcheck = service_healthchecks.get(service_name, None)
    if healthcheck is None:
        return
    click.secho(f"> Checking container health '{service_name}'", fg='yellow')

    def hc() -> None:
        if False:
            print('Hello World!')
        healthcheck.check(options)
    try:
        run_with_retries(hc, healthcheck.retries, healthcheck.timeout, f"Health check for '{service_name}' failed")
        click.secho(f"  > '{service_name}' is healthy", fg='green')
    except subprocess.CalledProcessError:
        click.secho(f"  > '{service_name}' is not healthy", fg='red')
        raise

def run_with_retries(cmd: Callable[[], object], retries: int=3, timeout: int=5, message: str='Command failed') -> None:
    if False:
        print('Hello World!')
    for retry in range(1, retries + 1):
        try:
            cmd()
        except (subprocess.CalledProcessError, urllib.error.HTTPError, http.client.RemoteDisconnected):
            if retry == retries:
                raise
            else:
                click.secho(f'  > {message}, retrying in {timeout}s (attempt {retry + 1} of {retries})...', fg='yellow')
                time.sleep(timeout)
        else:
            return

def check_postgres(options: dict[str, Any]) -> None:
    if False:
        for i in range(10):
            print('nop')
    subprocess.run(('docker', 'exec', options['name'], 'pg_isready', '-U', 'postgres'), check=True, capture_output=True, text=True)

def check_rabbitmq(options: dict[str, Any]) -> None:
    if False:
        return 10
    subprocess.run(('docker', 'exec', options['name'], 'rabbitmq-diagnostics', '-q', 'ping'), check=True, capture_output=True, text=True)

def check_redis(options: dict[str, Any]) -> None:
    if False:
        return 10
    subprocess.run(('docker', 'exec', options['name'], 'redis-cli', 'ping'), check=True, capture_output=True, text=True)

def check_vroom(options: dict[str, Any]) -> None:
    if False:
        for i in range(10):
            print('nop')
    (port,) = options['ports'].values()
    urllib.request.urlopen(f'http://{port[0]}:{port[1]}/health', timeout=1)

def check_clickhouse(options: dict[str, Any]) -> None:
    if False:
        while True:
            i = 10
    port = options['ports']['8123/tcp']
    subprocess.run(('docker', 'exec', options['name'], 'wget', f'http://{port[0]}:{port[1]}/ping'), check=True, capture_output=True, text=True)

def check_kafka(options: dict[str, Any]) -> None:
    if False:
        return 10
    (port,) = options['ports'].values()
    subprocess.run(('docker', 'exec', options['name'], 'kafka-topics', '--bootstrap-server', f'{port[0]}:{port[1]}', '--list'), check=True, capture_output=True, text=True)

def check_symbolicator(options: dict[str, Any]) -> None:
    if False:
        while True:
            i = 10
    (port,) = options['ports'].values()
    subprocess.run(('docker', 'exec', options['name'], 'curl', f'http://{port[0]}:{port[1]}/healthcheck'), check=True, capture_output=True, text=True)

def python_call_url_prog(url: str) -> str:
    if False:
        print('Hello World!')
    return f"\nimport urllib.request\ntry:\n    req = urllib.request.urlopen({url!r}, timeout=1)\nexcept Exception as e:\n    raise SystemExit(f'service is not ready: {{e}}')\nelse:\n    print('service is ready!')\n"

def check_chartcuterie(options: dict[str, Any]) -> None:
    if False:
        print('Hello World!')
    internal_port = 9090
    port = options['ports'][f'{internal_port}/tcp']
    url = f'http://{port[0]}:{internal_port}/api/chartcuterie/healthcheck/live'
    subprocess.run(('docker', 'exec', options['name'], 'python3', '-uc', python_call_url_prog(url)), check=True, capture_output=True, text=True)

def check_snuba(options: dict[str, Any]) -> None:
    if False:
        i = 10
        return i + 15
    from django.conf import settings
    url = f'{settings.SENTRY_SNUBA}/health_envoy'
    subprocess.run(('docker', 'exec', options['name'], 'python3', '-uc', python_call_url_prog(url)), check=True, capture_output=True, text=True)

class ServiceHealthcheck(NamedTuple):
    check: Callable[[dict[str, Any]], None]
    retries: int = 3
    timeout: int = 5
service_healthchecks: dict[str, ServiceHealthcheck] = {'postgres': ServiceHealthcheck(check=check_postgres), 'rabbitmq': ServiceHealthcheck(check=check_rabbitmq), 'redis': ServiceHealthcheck(check=check_redis), 'clickhouse': ServiceHealthcheck(check=check_clickhouse), 'kafka': ServiceHealthcheck(check=check_kafka), 'vroom': ServiceHealthcheck(check=check_vroom), 'symbolicator': ServiceHealthcheck(check=check_symbolicator), 'chartcuterie': ServiceHealthcheck(check=check_chartcuterie), 'snuba': ServiceHealthcheck(check=check_snuba, retries=12, timeout=10)}