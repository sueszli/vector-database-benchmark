import os
import re
import shutil
import subprocess
import sys
from typing import Dict, Optional, Tuple
import click
import requests
from packaging.version import Version
from lightning.app.core.constants import LIGHTNING_APPS_PUBLIC_REGISTRY, LIGHTNING_COMPONENT_PUBLIC_REGISTRY
from lightning.app.utilities.app_helpers import Logger
logger = Logger(__name__)

@click.group(name='install')
def install() -> None:
    if False:
        i = 10
        return i + 15
    'Install Lightning AI selfresources.'
    pass

@install.command('app')
@click.argument('name', type=str)
@click.option('--yes', '-y', is_flag=True, help='disables prompt to ask permission to create env and run install cmds')
@click.option('--version', '-v', type=str, help="Specify the version to install. By default it uses 'latest'", default='latest', show_default=True)
@click.option('--overwrite', '-f', is_flag=True, default=False, help='When set, overwrite the app directory without asking if it already exists.')
def install_app(name: str, yes: bool, version: str, overwrite: bool=False) -> None:
    if False:
        while True:
            i = 10
    _install_app_command(name, yes, version, overwrite=overwrite)

@install.command('component')
@click.argument('name', type=str)
@click.option('--yes', '-y', is_flag=True, help='disables prompt to ask permission to create env and run install cmds')
@click.option('--version', '-v', type=str, help="Specify the version to install. By default it uses 'latest'", default='latest', show_default=True)
def install_component(name: str, yes: bool, version: str) -> None:
    if False:
        while True:
            i = 10
    _install_component_command(name, yes, version)

def _install_app_command(name: str, yes: bool, version: str, overwrite: bool=False) -> None:
    if False:
        return 10
    if 'github.com' in name:
        if version != 'latest':
            logger.warn(f"When installing from GitHub, only the 'latest' version is supported. The provided version ({version}) will be ignored.")
        return non_gallery_app(name, yes, overwrite=overwrite)
    return gallery_app(name, yes, version, overwrite=overwrite)

def _install_component_command(name: str, yes: bool, version: str, overwrite: bool=False) -> None:
    if False:
        for i in range(10):
            print('nop')
    if 'github.com' in name:
        if version != 'latest':
            logger.warn(f"When installing from GitHub, only the 'latest' version is supported. The provided version ({version}) will be ignored.")
        return non_gallery_component(name, yes)
    return gallery_component(name, yes, version)

def gallery_apps_and_components(name: str, yes_arg: bool, version_arg: str, cwd: Optional[str]=None, overwrite: bool=False) -> Optional[str]:
    if False:
        print('Hello World!')
    try:
        (org, app_or_component) = name.split('/')
    except Exception:
        return None
    (entry, kind) = _resolve_entry(name, version_arg)
    if kind == 'app':
        (source_url, git_url, folder_name, git_sha) = _show_install_app_prompt(entry, app_or_component, org, yes_arg, resource_type='app')
        _install_app_from_source(source_url, git_url, folder_name, cwd=cwd, overwrite=overwrite, git_sha=git_sha)
        return os.path.join(os.getcwd(), *entry['appEntrypointFile'].split('/'))
    if kind == 'component':
        (source_url, git_url, folder_name, git_sha) = _show_install_app_prompt(entry, app_or_component, org, yes_arg, resource_type='component')
        if '@' in git_url:
            git_url = git_url.split('git+')[1].split('@')[0]
        _install_app_from_source(source_url, git_url, folder_name, cwd=cwd, overwrite=overwrite, git_sha=git_sha)
        return os.path.join(os.getcwd(), *entry['entrypointFile'].split('/'))
    return None

def gallery_component(name: str, yes_arg: bool, version_arg: str, cwd: Optional[str]=None) -> str:
    if False:
        return 10
    (org, component) = _validate_name(name, resource_type='component', example='lightning/LAI-slack-component')
    registry_url = _resolve_component_registry()
    component_entry = _resolve_resource(registry_url, name=name, version_arg=version_arg, resource_type='component')
    git_url = _show_install_component_prompt(component_entry, component, org, yes_arg)
    _install_component_from_source(git_url)
    return os.path.join(os.getcwd(), component_entry['entrypointFile'])

def non_gallery_component(gh_url: str, yes_arg: bool, cwd: Optional[str]=None) -> None:
    if False:
        while True:
            i = 10
    git_url = _show_non_gallery_install_component_prompt(gh_url, yes_arg)
    _install_component_from_source(git_url)

def gallery_app(name: str, yes_arg: bool, version_arg: str, cwd: Optional[str]=None, overwrite: bool=False) -> str:
    if False:
        print('Hello World!')
    (org, app) = _validate_name(name, resource_type='app', example='lightning/quick-start')
    registry_url = _resolve_app_registry()
    app_entry = _resolve_resource(registry_url, name=name, version_arg=version_arg, resource_type='app')
    (source_url, git_url, folder_name, git_sha) = _show_install_app_prompt(app_entry, app, org, yes_arg, resource_type='app')
    _install_app_from_source(source_url, git_url, folder_name, cwd=cwd, overwrite=overwrite, git_sha=git_sha)
    return os.path.join(os.getcwd(), folder_name, app_entry['appEntrypointFile'])

def non_gallery_app(gh_url: str, yes_arg: bool, cwd: Optional[str]=None, overwrite: bool=False) -> None:
    if False:
        while True:
            i = 10
    (repo_url, folder_name) = _show_non_gallery_install_app_prompt(gh_url, yes_arg)
    _install_app_from_source(repo_url, repo_url, folder_name, cwd=cwd, overwrite=overwrite)

def _show_install_component_prompt(entry: Dict[str, str], component: str, org: str, yes_arg: bool) -> str:
    if False:
        while True:
            i = 10
    git_url = entry['gitUrl']
    if yes_arg:
        return git_url
    prompt = f'\n    ⚡ Installing Lightning component ⚡\n\n    component name : {component}\n    developer      : {org}\n\n    Installation runs the following command for you:\n\n    pip install {git_url}\n    '
    logger.info(prompt)
    try:
        value = input('\nPress enter to continue:   ')
        value = value.strip().lower()
        should_install = len(value) == 0 or value in {'y', 'yes', 1}
        if not should_install:
            raise KeyboardInterrupt()
        return git_url
    except KeyboardInterrupt:
        repo = entry['sourceUrl']
        raise SystemExit(f'\n        ⚡ Installation aborted! ⚡\n\n        Install the component yourself by visiting:\n        {repo}\n        ')

def _show_non_gallery_install_component_prompt(gh_url: str, yes_arg: bool) -> str:
    if False:
        for i in range(10):
            print('nop')
    if '.git@' not in gh_url:
        m = '\n        Error, your github url must be in the following format:\n        git+https://github.com/OrgName/repo-name.git@ALongCommitSHAString\n\n        Example:\n        git+https://github.com/Lightning-AI/LAI-slack-messenger.git@14f333456ffb6758bd19458e6fa0bf12cf5575e1\n        '
        raise SystemExit(m)
    developer = gh_url.split('/')[3]
    component_name = gh_url.split('/')[4].split('.git')[0]
    repo_url = re.search('git\\+(.*).git', gh_url).group(1)
    if yes_arg:
        return gh_url
    prompt = f'\n    ⚡ Installing Lightning component ⚡\n\n    component name : {component_name}\n    developer      : {developer}\n\n    ⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡\n    WARNING: this is NOT an official Lightning Gallery component\n    Install at your own risk\n    ⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡\n\n    Installation runs the following command for you:\n\n    pip install {gh_url}\n    '
    logger.info(prompt)
    try:
        value = input('\nPress enter to continue:   ')
        value = value.strip().lower()
        should_install = len(value) == 0 or value in {'y', 'yes', 1}
        if not should_install:
            raise KeyboardInterrupt()
        return gh_url
    except KeyboardInterrupt:
        raise SystemExit(f'\n        ⚡ Installation aborted! ⚡\n\n        Install the component yourself by visiting:\n        {repo_url}\n        ')

def _show_install_app_prompt(entry: Dict[str, str], app: str, org: str, yes_arg: bool, resource_type: str) -> Tuple[str, str, str, Optional[str]]:
    if False:
        while True:
            i = 10
    source_url = entry['sourceUrl']
    full_git_url = entry['gitUrl']
    git_url_parts = full_git_url.split('#ref=')
    git_url = git_url_parts[0]
    git_sha = git_url_parts[1] if len(git_url_parts) == 2 else None
    folder_name = source_url.split('/')[-1]
    if yes_arg:
        return (source_url, git_url, folder_name, git_sha)
    prompt = f'\n    ⚡ Installing Lightning {resource_type} ⚡\n\n    {resource_type} name : {app}\n    developer: {org}\n\n    Installation creates and runs the following commands for you:\n\n    git clone {source_url}\n    cd {folder_name}\n    pip install -r requirements.txt\n    pip install -e .\n    '
    logger.info(prompt)
    try:
        value = input('\nPress enter to continue:   ')
        value = value.strip().lower()
        should_install = len(value) == 0 or value in {'y', 'yes', 1}
        if not should_install:
            raise KeyboardInterrupt()
        return (source_url, git_url, folder_name, git_sha)
    except KeyboardInterrupt:
        repo = entry['sourceUrl']
        raise SystemExit(f'\n        ⚡ Installation aborted! ⚡\n\n        Install the {resource_type} yourself by visiting:\n        {repo}\n        ')

def _show_non_gallery_install_app_prompt(gh_url: str, yes_arg: bool) -> Tuple[str, str]:
    if False:
        while True:
            i = 10
    try:
        if gh_url.endswith('.git'):
            folder_name = gh_url.split('/')[-1]
            folder_name = folder_name[:-4]
        else:
            folder_name = gh_url.split('/')[-1]
        org = re.search('github.com\\/(.*)\\/', gh_url).group(1)
    except Exception:
        raise SystemExit("\n        Your github url is not supported. Here's the supported format:\n        https://github.com/YourOrgName/your-repo-name\n\n        Example:\n        https://github.com/Lightning-AI/lightning\n        ")
    if yes_arg:
        return (gh_url, folder_name)
    prompt = f'\n    ⚡ Installing Lightning app ⚡\n\n    app source : {gh_url}\n    developer  : {org}\n\n    ⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡\n    WARNING: this is NOT an official Lightning Gallery app\n    Install at your own risk\n    ⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡⚡\n\n    Installation creates and runs the following commands for you:\n\n    git clone {gh_url}\n    cd {folder_name}\n    pip install -r requirements.txt\n    pip install -e .\n    '
    logger.info(prompt)
    try:
        value = input('\nPress enter to continue:   ')
        value = value.strip().lower()
        should_install = len(value) == 0 or value in {'y', 'yes', 1}
        if not should_install:
            raise KeyboardInterrupt()
        return (gh_url, folder_name)
    except KeyboardInterrupt:
        raise SystemExit(f'\n        ⚡ Installation aborted! ⚡\n\n        Install the app yourself by visiting {gh_url}\n        ')

def _validate_name(name: str, resource_type: str, example: str) -> Tuple[str, str]:
    if False:
        return 10
    try:
        (org, resource) = name.split('/')
    except Exception:
        raise SystemExit(f'\n        {resource_type} name format must have organization/{resource_type}-name\n\n        Examples:\n        {example}\n        user/{resource_type}-name\n\n        You passed in: {name}\n        ')
    return (org, resource)

def _resolve_entry(name, version_arg) -> Tuple[Optional[Dict], Optional[str]]:
    if False:
        i = 10
        return i + 15
    entry = None
    kind = None
    registry_url = _resolve_app_registry()
    entry = _resolve_resource(registry_url, name=name, version_arg=version_arg, resource_type='app', raise_error=False)
    if not entry:
        registry_url = _resolve_component_registry()
        entry = _resolve_resource(registry_url, name=name, version_arg=version_arg, resource_type='component', raise_error=False)
        kind = 'component' if entry else None
    else:
        kind = 'app'
    return (entry, kind)

def _resolve_resource(registry_url: str, name: str, version_arg: str, resource_type: str, raise_error: bool=True) -> Dict[str, str]:
    if False:
        for i in range(10):
            print('nop')
    gallery_entries = []
    try:
        response = requests.get(registry_url)
        data = response.json()
        if resource_type == 'app':
            gallery_entries = [a for a in data['apps'] if a['canDownloadSourceCode']]
        elif resource_type == 'component':
            gallery_entries = data['components']
    except requests.ConnectionError:
        sys.tracebacklimit = 0
        raise SystemError(f'\n        Network connection error, could not load list of available Lightning {resource_type}s.\n\n        Try again when you have a network connection!\n        ')
    entries = []
    all_versions = []
    for x in gallery_entries:
        if name == x['name']:
            entries.append(x)
            all_versions.append(x['version'])
    if len(entries) == 0:
        if raise_error:
            raise SystemExit(f"{resource_type}: '{name}' is not available on ⚡ Lightning AI ⚡")
        return None
    entry = None
    if version_arg == 'latest':
        entry = max(entries, key=lambda app: Version(app['version']))
    else:
        for e in entries:
            if e['version'] == version_arg:
                entry = e
                break
    if entry is None and raise_error:
        if raise_error:
            raise Exception(f"{resource_type}: 'Version {version_arg} for {name}' is not available on ⚡ Lightning AI ⚡. Here is the list of all availables versions:{os.linesep}{os.linesep.join(all_versions)}")
        return None
    return entry

def _install_with_env(repo_url: str, folder_name: str, cwd: Optional[str]=None) -> None:
    if False:
        while True:
            i = 10
    if not cwd:
        cwd = os.getcwd()
    logger.info(f'⚡ RUN: git clone {repo_url}')
    subprocess.call(['git', 'clone', repo_url])
    os.chdir(f'{folder_name}')
    cwd = os.getcwd()
    logger.info(f'⚡ CREATE: virtual env at {cwd}')
    subprocess.call(['python', '-m', 'venv', cwd])
    logger.info('⚡ RUN: install requirements (pip install -r requirements.txt)')
    subprocess.call('source bin/activate && pip install -r requirements.txt', shell=True)
    logger.info('⚡ RUN: setting up project (pip install -e .)')
    subprocess.call('source bin/activate && pip install -e .', shell=True)
    m = f'\n    ⚡ Installed! ⚡ to use your app\n        go into the folder: cd {folder_name}\n    activate the environment: source bin/activate\n                run the app: lightning run app [the_app_file.py]\n    '
    logger.info(m)

def _install_app_from_source(source_url: str, git_url: str, folder_name: str, cwd: Optional[str]=None, overwrite: bool=False, git_sha: Optional[str]=None) -> None:
    if False:
        while True:
            i = 10
    'Installing lighting app from the `git_url`\n\n    Args:\n        source_url:\n            source repo url without any tokens and params, this param is used only for displaying\n        git_url:\n            repo url that is used to clone, this can contain tokens\n        folder_name:\n            where to clone the repo ?\n        cwd:\n            Working director. If not specified, current working directory is used.\n        overwrite:\n            If true, overwrite the app directory without asking if it already exists\n        git_sha:\n            The git_sha for checking out the git repo of the app.\n\n    '
    if not cwd:
        cwd = os.getcwd()
    destination = os.path.join(cwd, folder_name)
    if os.path.exists(destination):
        if not overwrite:
            raise SystemExit(f'Folder {folder_name} exists, please delete it and try again, or force to overwrite the existing folder by passing `--overwrite`.')
        shutil.rmtree(destination)
    logger.info(f'⚡ RUN: git clone {source_url}')
    try:
        subprocess.check_output(['git', 'clone', git_url], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as ex:
        if 'Repository not found' in str(ex.output):
            raise SystemExit(f"\n            Looks like the github url was not found or doesn't exist. Do you have a typo?\n            {source_url}\n            ")
        raise Exception(ex)
    os.chdir(f'{folder_name}')
    cwd = os.getcwd()
    try:
        if git_sha:
            subprocess.check_output(['git', 'checkout', git_sha], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as ex:
        if 'did not match any' in str(ex.output):
            raise SystemExit("Looks like the git SHA is not valid or doesn't exist in app repo.")
        raise Exception(ex)
    logger.info('⚡ RUN: install requirements (pip install -r requirements.txt)')
    subprocess.call('pip install -r requirements.txt', shell=True)
    logger.info('⚡ RUN: setting up project (pip install -e .)')
    subprocess.call('pip install -e .', shell=True)
    m = f'\n    ⚡ Installed! ⚡ to use your app:\n\n    cd {folder_name}\n    lightning run app app.py\n    '
    logger.info(m)

def _install_component_from_source(git_url: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    logger.info('⚡ RUN: pip install')
    out = subprocess.check_output(['pip', 'install', git_url])
    possible_success_message = [x for x in str(out).split('\\n') if 'Successfully installed' in x]
    if len(possible_success_message) > 0:
        uninstall_step = possible_success_message[0]
        uninstall_step = re.sub('Successfully installed', '', uninstall_step).strip()
        uninstall_step = re.sub('-0.0.0', '', uninstall_step).strip()
        m = '\n        ⚡ Installed! ⚡\n\n        to use your component:\n        from the_component import TheClass\n\n        make sure to add this entry to your Lightning APP requirements.txt file:\n        {git_url}\n\n        if you want to uninstall, run this command:\n        pip uninstall {uninstall_step}\n        '
        logger.info(m)

def _resolve_app_registry() -> str:
    if False:
        for i in range(10):
            print('nop')
    return os.environ.get('LIGHTNING_APP_REGISTRY', LIGHTNING_APPS_PUBLIC_REGISTRY)

def _resolve_component_registry() -> str:
    if False:
        return 10
    return os.environ.get('LIGHTNING_COMPONENT_REGISTRY', LIGHTNING_COMPONENT_PUBLIC_REGISTRY)