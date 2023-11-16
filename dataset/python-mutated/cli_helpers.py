import contextlib
import functools
import json
import os
import re
import subprocess
import sys
from typing import Dict, Optional
import arrow
import click
import packaging
import requests
import rich
from lightning_cloud.openapi import Externalv1LightningappInstance
from lightning.app import __package_name__, __version__
from lightning.app.core.constants import APP_SERVER_PORT
from lightning.app.utilities.app_helpers import Logger
from lightning.app.utilities.cloud import _get_project
from lightning.app.utilities.network import LightningClient
logger = Logger(__name__)

def _format_input_env_variables(env_list: tuple) -> Dict[str, str]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Args:\n        env_list:\n           List of str for the env variables, e.g. ['foo=bar', 'bla=bloz']\n\n    Returns:\n        Dict of the env variables with the following format\n            key: env variable name\n            value: env variable value\n    "
    env_vars_dict = {}
    for env_str in env_list:
        var_parts = env_str.split('=')
        if len(var_parts) != 2 or not var_parts[0]:
            raise Exception(f'Invalid format of environment variable {env_str}, please ensure that the variable is in the format e.g. foo=bar.')
        (var_name, value) = var_parts
        if var_name in env_vars_dict:
            raise Exception(f"Environment variable '{var_name}' is duplicated. Please only include it once.")
        if not re.match('[0-9a-zA-Z_]+', var_name):
            raise ValueError(f"Environment variable '{var_name}' is not a valid name. It is only allowed to contain digits 0-9, letters A-Z, a-z and _ (underscore).")
        env_vars_dict[var_name] = value
    return env_vars_dict

def _is_url(id: Optional[str]) -> bool:
    if False:
        print('Hello World!')
    if isinstance(id, str) and (id.startswith('https://') or id.startswith('http://')):
        return True
    return False

def _get_metadata_from_openapi(paths: Dict, path: str):
    if False:
        i = 10
        return i + 15
    parameters = paths[path]['post'].get('parameters', {})
    tag = paths[path]['post'].get('tags', [None])[0]
    cls_path = paths[path]['post'].get('cls_path', None)
    cls_name = paths[path]['post'].get('cls_name', None)
    description = paths[path]['post'].get('description', None)
    requirements = paths[path]['post'].get('requirements', None)
    app_info = paths[path]['post'].get('app_info', None)
    metadata = {'tag': tag, 'parameters': {}}
    if cls_path:
        metadata['cls_path'] = cls_path
    if cls_name:
        metadata['cls_name'] = cls_name
    if description:
        metadata['description'] = description
    if description:
        metadata['requirements'] = requirements
    if app_info:
        metadata['app_info'] = app_info
    if not parameters:
        return metadata
    metadata['parameters'].update({d['name']: d['schema']['type'] for d in parameters})
    return metadata

def _extract_command_from_openapi(openapi_resp: Dict) -> Dict[str, Dict[str, str]]:
    if False:
        i = 10
        return i + 15
    command_paths = [p for p in openapi_resp['paths'] if p.startswith('/command/')]
    return {p.replace('/command/', ''): _get_metadata_from_openapi(openapi_resp['paths'], p) for p in command_paths}

def _get_app_display_name(app: Externalv1LightningappInstance) -> str:
    if False:
        while True:
            i = 10
    return getattr(app, 'display_name', None) or app.name

class _LightningAppOpenAPIRetriever:

    def __init__(self, app_id_or_name_or_url: Optional[str], use_cache: bool=False):
        if False:
            while True:
                i = 10
        'This class encapsulates the logic to collect the openapi.json file from the app to use the CLI Commands.\n\n        Arguments:\n            app_id_or_name_or_url: An identified for the app.\n            use_cache: Whether to load the openapi spec from the cache.\n\n        '
        self.app_id_or_name_or_url = app_id_or_name_or_url
        self.url = None
        self.openapi = None
        self.api_commands = None
        self.app_id = None
        self.app_name = None
        home = os.path.expanduser('~')
        if use_cache:
            cache_openapi = os.path.join(home, '.lightning', 'lightning_connection', 'commands', 'openapi.json')
            if os.path.exists(cache_openapi):
                with open(cache_openapi) as f:
                    self.openapi = json.load(f)
                self.api_commands = _extract_command_from_openapi(self.openapi)
        if not self.api_commands:
            self._collect_open_api_json()
            if self.openapi:
                self.api_commands = _extract_command_from_openapi(self.openapi)

    def is_alive(self) -> bool:
        if False:
            return 10
        'Returns whether the Lightning App Rest API is available.'
        if self.url is None:
            self._maybe_find_url()
        if self.url is None:
            return False
        resp = requests.get(self.url)
        return resp.status_code == 200

    def _maybe_find_url(self):
        if False:
            print('Hello World!')
        'Tries to resolve the app url from the provided `app_id_or_name_or_url`.'
        if _is_url(self.app_id_or_name_or_url):
            self.url = self.app_id_or_name_or_url
            assert self.url
            return
        if self.app_id_or_name_or_url is None:
            url = f'http://localhost:{APP_SERVER_PORT}'
            resp = requests.get(f'{self.url}/openapi.json')
            if resp.status_code == 200:
                self.url = url
                return
        app = self._maybe_find_matching_cloud_app()
        if app:
            self.url = app.status.url

    def _maybe_find_matching_cloud_app(self):
        if False:
            for i in range(10):
                print('nop')
        'Tries to resolve the app url from the provided `app_id_or_name_or_url`.'
        client = LightningClient(retry=False)
        project = _get_project(client)
        list_apps = client.lightningapp_instance_service_list_lightningapp_instances(project_id=project.project_id)
        app_names = [_get_app_display_name(lit_app) for lit_app in list_apps.lightningapps]
        if not self.app_id_or_name_or_url:
            print(f'ERROR: Provide an application name, id or url with --app_id=X. Found {app_names}')
            sys.exit(0)
        for app in list_apps.lightningapps:
            if app.id == self.app_id_or_name_or_url or _get_app_display_name(app) == self.app_id_or_name_or_url:
                if app.status.url == '':
                    print('The application is starting. Try in a few moments.')
                    sys.exit(0)
                return app
        return None

    def _collect_open_api_json(self):
        if False:
            for i in range(10):
                print('nop')
        'This function is used to retrieve the current url associated with an id.'
        if _is_url(self.app_id_or_name_or_url):
            self.url = self.app_id_or_name_or_url
            assert self.url
            resp = requests.get(self.url + '/openapi.json')
            if resp.status_code != 200:
                print(f"ERROR: The server didn't process the request properly. Found {resp.json()}")
                sys.exit(0)
            self.openapi = resp.json()
            return
        if self.app_id_or_name_or_url is None:
            with contextlib.suppress(requests.exceptions.ConnectionError):
                self.url = f'http://localhost:{APP_SERVER_PORT}'
                resp = requests.get(f'{self.url}/openapi.json')
                if resp.status_code != 200:
                    raise Exception(f"The server didn't process the request properly. Found {resp.json()}")
                self.openapi = resp.json()
        else:
            app = self._maybe_find_matching_cloud_app()
            if app:
                if app.status.url == '':
                    raise Exception('The application is starting. Try in a few moments.')
                resp = requests.get(app.status.url + '/openapi.json')
                if resp.status_code != 200:
                    raise Exception("The server didn't process the request properly. Try once your application is ready.")
                self.url = app.status.url
                self.openapi = resp.json()
                self.app_id = app.id
                self.app_name = _get_app_display_name(app)

def _arrow_time_callback(_ctx: 'click.core.Context', _param: 'click.core.Option', value: str, arw_now=arrow.utcnow()) -> arrow.Arrow:
    if False:
        print('Hello World!')
    try:
        return arw_now.dehumanize(value)
    except ValueError:
        try:
            return arrow.get(value)
        except (ValueError, TypeError):
            raise click.ClickException(f'cannot parse time {value}')

@functools.lru_cache(maxsize=1)
def _get_newer_version() -> Optional[str]:
    if False:
        return 10
    'Check PyPI for newer versions of ``lightning``, returning the newest version if different from the current or\n    ``None`` otherwise.'
    if packaging.version.parse(__version__).is_prerelease:
        return None
    try:
        response = requests.get(f'https://pypi.org/pypi/{__package_name__}/json')
        response_json = response.json()
        releases = response_json['releases']
        if __version__ not in releases:
            return None
        latest_version = response_json['info']['version']
        parsed_version = packaging.version.parse(latest_version)
        is_invalid = response_json['info']['yanked'] or parsed_version.is_devrelease or parsed_version.is_prerelease
        return None if __version__ == latest_version or is_invalid else latest_version
    except Exception:
        return None

def _redirect_command(executable: str):
    if False:
        i = 10
        return i + 15
    'Redirect the current lightning CLI call to the given executable.'
    subprocess.run([executable, '-m', 'lightning'] + sys.argv[1:], env=os.environ)
    sys.exit()

def _check_version_and_upgrade():
    if False:
        i = 10
        return i + 15
    'Checks that the current version of ``lightning`` is the latest on PyPI.\n\n    If not, prompt the user to upgrade ``lightning`` for them and re-run the current call in the new version.\n\n    '
    new_version = _get_newer_version()
    if new_version:
        prompt = f'A newer version of {__package_name__} is available ({new_version}). Would you like to upgrade?'
        if click.confirm(prompt, default=True):
            command = f'pip install {__package_name__}=={new_version}'
            logger.info(f'⚡ RUN: {command}')
            subprocess.run([sys.executable, '-m'] + command.split(' '), check=True)
            _redirect_command(sys.executable)
    return

def _check_environment_and_redirect():
    if False:
        for i in range(10):
            print('nop')
    'Checks that the current ``sys.executable`` is the same as the executable resolved from the current environment.\n\n    If not, this utility tries to redirect the ``lightning`` call to the environment executable (prompting the user to\n    install lightning for them there if needed).\n\n    '
    process = subprocess.run(['python', '-c', 'import sys; print(sys.executable)'], capture_output=True, env=os.environ, check=True)
    env_executable = os.path.realpath(process.stdout.decode().strip())
    sys_executable = os.path.realpath(sys.executable)
    if env_executable.lower() != sys_executable.lower():
        logger.info('Lightning is running from outside your current environment. Switching to your current environment.')
        process = subprocess.run([env_executable, '-m', 'lightning', '--version'], capture_output=True, text=True)
        if 'No module named lightning' in process.stderr:
            prompt = f'The {__package_name__} package is not installed. Would you like to install it? [Y/n (exit)]'
            if click.confirm(prompt, default=True, show_default=False):
                command = f'pip install {__package_name__}'
                logger.info(f'⚡ RUN: {command}')
                subprocess.run([env_executable, '-m'] + command.split(' '), check=True)
            else:
                sys.exit()
        _redirect_command(env_executable)
    return

def _error_and_exit(msg: str) -> None:
    if False:
        print('Hello World!')
    rich.print(f'[red]ERROR[/red]: {msg}')
    sys.exit(0)