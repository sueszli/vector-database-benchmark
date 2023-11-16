"""Setup.py for the Provider packages of Airflow project."""
from __future__ import annotations
import glob
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from collections import namedtuple
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from pathlib import Path
from shutil import copyfile
from typing import Any, Generator, Iterable, NamedTuple
import jinja2
import jsonschema
import rich_click as click
import semver as semver
from black import Mode, TargetVersion, format_str, parse_pyproject_toml
from packaging.version import Version
from rich.console import Console
from rich.syntax import Syntax
from yaml import safe_load
ALL_PYTHON_VERSIONS = ['3.8', '3.9', '3.10', '3.11']
MIN_AIRFLOW_VERSION = '2.5.0'
INITIAL_CHANGELOG_CONTENT = '\n .. Licensed to the Apache Software Foundation (ASF) under one\n    or more contributor license agreements.  See the NOTICE file\n    distributed with this work for additional information\n    regarding copyright ownership.  The ASF licenses this file\n    to you under the Apache License, Version 2.0 (the\n    "License"); you may not use this file except in compliance\n    with the License.  You may obtain a copy of the License at\n\n ..   http://www.apache.org/licenses/LICENSE-2.0\n\n .. Unless required by applicable law or agreed to in writing,\n    software distributed under the License is distributed on an\n    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n    KIND, either express or implied.  See the License for the\n    specific language governing permissions and limitations\n    under the License.\n\n.. NOTE TO CONTRIBUTORS:\n   Please, only add notes to the Changelog just below the "Changelog" header when there\n   are some breaking changes and you want to add an explanation to the users on how they are supposed\n   to deal with them. The changelog is updated and maintained semi-automatically by release manager.\n\n``{{ package_name }}``\n\nChangelog\n---------\n\n1.0.0\n.....\n\nInitial version of the provider.\n'
HTTPS_REMOTE = 'apache-https-for-providers'
HEAD_OF_HTTPS_REMOTE = f'{HTTPS_REMOTE}'
MY_DIR_PATH = Path(__file__).parent
AIRFLOW_SOURCES_ROOT_PATH = MY_DIR_PATH.parents[1]
AIRFLOW_PATH = AIRFLOW_SOURCES_ROOT_PATH / 'airflow'
DIST_PATH = AIRFLOW_SOURCES_ROOT_PATH / 'dist'
PROVIDERS_PATH = AIRFLOW_PATH / 'providers'
DOCUMENTATION_PATH = AIRFLOW_SOURCES_ROOT_PATH / 'docs'
DEPENDENCIES_JSON_FILE_PATH = AIRFLOW_SOURCES_ROOT_PATH / 'generated' / 'provider_dependencies.json'
TARGET_PROVIDER_PACKAGES_PATH = AIRFLOW_SOURCES_ROOT_PATH / 'provider_packages'
GENERATED_AIRFLOW_PATH = TARGET_PROVIDER_PACKAGES_PATH / 'airflow'
GENERATED_PROVIDERS_PATH = GENERATED_AIRFLOW_PATH / 'providers'
PROVIDER_RUNTIME_DATA_SCHEMA_PATH = AIRFLOW_SOURCES_ROOT_PATH / 'airflow' / 'provider_info.schema.json'
CROSS_PROVIDERS_DEPS = 'cross-providers-deps'
DEPS = 'deps'
sys.path.insert(0, str(AIRFLOW_SOURCES_ROOT_PATH))
ALL_DEPENDENCIES = json.loads(DEPENDENCIES_JSON_FILE_PATH.read_text())
from setup import PREINSTALLED_PROVIDERS, ALL_PROVIDERS
logger = logging.getLogger(__name__)
PY3 = sys.version_info[0] == 3
console = Console(width=400, color_system='standard')

class PluginInfo(NamedTuple):
    name: str
    package_name: str
    class_name: str

class ProviderPackageDetails(NamedTuple):
    provider_package_id: str
    full_package_name: str
    pypi_package_name: str
    source_provider_package_path: str
    documentation_provider_package_path: Path
    provider_description: str
    versions: list[str]
    excluded_python_versions: list[str]
    plugins: list[PluginInfo]
    removed: bool

class EntityType(Enum):
    Operators = 'Operators'
    Transfers = 'Transfers'
    Sensors = 'Sensors'
    Hooks = 'Hooks'
    Secrets = 'Secrets'

@click.group(context_settings={'help_option_names': ['-h', '--help'], 'max_content_width': 500})
def cli():
    if False:
        print('Hello World!')
    ...
option_skip_tag_check = click.option('--skip-tag-check/--no-skip-tag-check', default=False, is_flag=True, help='Skip checking if the tag already exists in the remote repository')
option_git_update = click.option('--git-update/--no-git-update', default=True, is_flag=True, help=f"If the git remote {HTTPS_REMOTE} already exists, don't try to update it")
option_package_format = click.option('--package-format', type=click.Choice(['wheel', 'sdist', 'both']), help='Format of packages.', default='wheel', show_default=True, envvar='PACKAGE_FORMAT')
option_version_suffix = click.option('--version-suffix', metavar='suffix', help=textwrap.dedent('\n        adds version suffix to version of the packages.\n        only useful when generating rc candidates for pypi.'))
option_verbose = click.option('--verbose', is_flag=True, help='Print verbose information about performed steps')
argument_package_id = click.argument('package_id')

@contextmanager
def with_group(title: str) -> Generator[None, None, None]:
    if False:
        return 10
    '\n    If used in GitHub Action, creates an expandable group in the GitHub Action log.\n    Otherwise, display simple text groups.\n\n    For more information, see:\n    https://docs.github.com/en/free-pro-team@latest/actions/reference/workflow-commands-for-github-actions#grouping-log-lines\n    '
    if os.environ.get('GITHUB_ACTIONS', 'false') != 'true':
        console.print('#' * 10 + ' [bright_blue]' + title + '[/] ' + '#' * 10)
        yield
        return
    console.print(f'::group::[bright_blue]{title}[/]')
    yield
    console.print('::endgroup::')

def get_source_airflow_folder() -> str:
    if False:
        print('Hello World!')
    '\n    Returns source directory for whole airflow (from the main airflow project).\n\n    :return: the folder path\n    '
    return os.path.abspath(AIRFLOW_SOURCES_ROOT_PATH)

def get_source_providers_folder() -> str:
    if False:
        while True:
            i = 10
    '\n    Returns source directory for providers (from the main airflow project).\n\n    :return: the folder path\n    '
    return os.path.join(get_source_airflow_folder(), 'airflow', 'providers')

def get_target_folder() -> str:
    if False:
        return 10
    '\n    Returns target directory for providers (in the provider_packages folder)\n\n    :return: the folder path\n    '
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'provider_packages'))

def get_target_providers_folder() -> str:
    if False:
        while True:
            i = 10
    '\n    Returns target directory for providers (in the provider_packages folder)\n\n    :return: the folder path\n    '
    return os.path.abspath(os.path.join(get_target_folder(), 'airflow', 'providers'))

def get_target_providers_package_folder(provider_package_id: str) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Returns target package folder based on package_id\n\n    :return: the folder path\n    '
    return os.path.join(get_target_providers_folder(), *provider_package_id.split('.'))

def get_pip_package_name(provider_package_id: str) -> str:
    if False:
        print('Hello World!')
    '\n    Returns PIP package name for the package id.\n\n    :param provider_package_id: id of the package\n    :return: the name of pip package\n    '
    return 'apache-airflow-providers-' + provider_package_id.replace('.', '-')

def get_wheel_package_name(provider_package_id: str) -> str:
    if False:
        print('Hello World!')
    '\n    Returns Wheel package name for the package id.\n\n    :param provider_package_id: id of the package\n    :return: the name of pip package\n    '
    return 'apache_airflow_providers_' + provider_package_id.replace('.', '_')

def get_install_requirements(provider_package_id: str, version_suffix: str) -> str:
    if False:
        return 10
    '\n    Returns install requirements for the package.\n\n    :param provider_package_id: id of the provider package\n    :param version_suffix: optional version suffix for packages\n\n    :return: install requirements of the package\n    '

    def apply_version_suffix(install_clause: str) -> str:
        if False:
            i = 10
            return i + 15
        if install_clause.startswith('apache-airflow') and '>=' in install_clause and (version_suffix != ''):
            return install_clause + '.dev0'
        return install_clause
    if provider_package_id in get_removed_provider_ids():
        provider_info = get_provider_info_from_provider_yaml(provider_package_id)
        dependencies = provider_info['dependencies']
    else:
        dependencies = ALL_DEPENDENCIES[provider_package_id][DEPS]
    install_requires = [apply_version_suffix(clause) for clause in dependencies]
    return ''.join((f'\n    {ir}' for ir in install_requires))

def get_setup_requirements() -> str:
    if False:
        while True:
            i = 10
    '\n    Returns setup requirements (common for all package for now).\n    :return: setup requirements\n    '
    return '\n    setuptools\n    wheel\n'

def get_package_extras(provider_package_id: str) -> dict[str, list[str]]:
    if False:
        while True:
            i = 10
    '\n    Finds extras for the package specified.\n\n    :param provider_package_id: id of the package\n    '
    if provider_package_id == 'providers':
        return {}
    if provider_package_id in get_removed_provider_ids():
        return {}
    extras_dict: dict[str, list[str]] = {module: [get_pip_package_name(module)] for module in ALL_DEPENDENCIES[provider_package_id][CROSS_PROVIDERS_DEPS]}
    provider_yaml_dict = get_provider_yaml(provider_package_id)
    additional_extras = provider_yaml_dict.get('additional-extras')
    if additional_extras:
        for entry in additional_extras:
            name = entry['name']
            dependencies = entry['dependencies']
            if name in extras_dict:
                existing_dependencies = set(extras_dict[name])
                for new_dependency in dependencies:
                    for dependency in existing_dependencies:
                        if new_dependency.startswith(dependency):
                            extras_dict[name].remove(dependency)
                            break
                    extras_dict[name].append(new_dependency)
            else:
                extras_dict[name] = dependencies
    return extras_dict

def render_template(template_name: str, context: dict[str, Any], extension: str, autoescape: bool=True, keep_trailing_newline: bool=False) -> str:
    if False:
        while True:
            i = 10
    '\n    Renders template based on its name. Reads the template from <name>_TEMPLATE.md.jinja2 in current dir.\n    :param template_name: name of the template to use\n    :param context: Jinja2 context\n    :param extension: Target file extension\n    :param autoescape: Whether to autoescape HTML\n    :param keep_trailing_newline: Whether to keep the newline in rendered output\n    :return: rendered template\n    '
    import jinja2
    template_loader = jinja2.FileSystemLoader(searchpath=MY_DIR_PATH)
    template_env = jinja2.Environment(loader=template_loader, undefined=jinja2.StrictUndefined, autoescape=autoescape, keep_trailing_newline=keep_trailing_newline)
    template = template_env.get_template(f'{template_name}_TEMPLATE{extension}.jinja2')
    content: str = template.render(context)
    return content
PR_PATTERN = re.compile('.*\\(#(\\d+)\\)')

class Change(NamedTuple):
    """Stores details about commits"""
    full_hash: str
    short_hash: str
    date: str
    version: str
    message: str
    message_without_backticks: str
    pr: str | None

def get_change_from_line(line: str, version: str):
    if False:
        while True:
            i = 10
    split_line = line.split(' ', maxsplit=3)
    message = split_line[3]
    pr = None
    pr_match = PR_PATTERN.match(message)
    if pr_match:
        pr = pr_match.group(1)
    return Change(full_hash=split_line[0], short_hash=split_line[1], date=split_line[2], version=version, message=message, message_without_backticks=message.replace('`', "'").replace('&39;', "'"), pr=pr)

def convert_pip_requirements_to_table(requirements: Iterable[str], markdown: bool=True) -> str:
    if False:
        while True:
            i = 10
    '\n    Converts PIP requirement list to a Markdown table.\n    :param requirements: requirements list\n    :param markdown: if True, Markdown format is used else rst\n    :return: formatted table\n    '
    from tabulate import tabulate
    headers = ['PIP package', 'Version required']
    table_data = []
    for dependency in requirements:
        found = re.match('(^[^<=>~]*)([^<=>~]?.*)$', dependency)
        if found:
            package = found.group(1)
            version_required = found.group(2)
            if version_required != '':
                version_required = f'`{version_required}`' if markdown else f'``{version_required}``'
            table_data.append((f'`{package}`' if markdown else f'``{package}``', version_required))
        else:
            table_data.append((dependency, ''))
    return tabulate(table_data, headers=headers, tablefmt='pipe' if markdown else 'rst')

def convert_cross_package_dependencies_to_table(cross_package_dependencies: list[str], markdown: bool=True) -> str:
    if False:
        print('Hello World!')
    '\n    Converts cross-package dependencies to a Markdown table\n    :param cross_package_dependencies: list of cross-package dependencies\n    :param markdown: if True, Markdown format is used else rst\n    :return: formatted table\n    '
    from tabulate import tabulate
    headers = ['Dependent package', 'Extra']
    table_data = []
    prefix = 'apache-airflow-providers-'
    base_url = 'https://airflow.apache.org/docs/'
    for dependency in cross_package_dependencies:
        pip_package_name = f"{prefix}{dependency.replace('.', '-')}"
        url_suffix = f"{dependency.replace('.', '-')}"
        if markdown:
            url = f'[{pip_package_name}]({base_url}{url_suffix})'
        else:
            url = f'`{pip_package_name} <{base_url}{prefix}{url_suffix}>`_'
        table_data.append((url, f'`{dependency}`' if markdown else f'``{dependency}``'))
    return tabulate(table_data, headers=headers, tablefmt='pipe' if markdown else 'rst')
LICENCE = '<!--\n Licensed to the Apache Software Foundation (ASF) under one\n or more contributor license agreements.  See the NOTICE file\n distributed with this work for additional information\n regarding copyright ownership.  The ASF licenses this file\n to you under the Apache License, Version 2.0 (the\n "License"); you may not use this file except in compliance\n with the License.  You may obtain a copy of the License at\n\n   http://www.apache.org/licenses/LICENSE-2.0\n\n Unless required by applicable law or agreed to in writing,\n software distributed under the License is distributed on an\n "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n KIND, either express or implied.  See the License for the\n specific language governing permissions and limitations\n under the License.\n -->\n'
LICENCE_RST = '\n.. Licensed to the Apache Software Foundation (ASF) under one\n   or more contributor license agreements.  See the NOTICE file\n   distributed with this work for additional information\n   regarding copyright ownership.  The ASF licenses this file\n   to you under the Apache License, Version 2.0 (the\n   "License"); you may not use this file except in compliance\n   with the License.  You may obtain a copy of the License at\n\n..   http://www.apache.org/licenses/LICENSE-2.0\n\n.. Unless required by applicable law or agreed to in writing,\n   software distributed under the License is distributed on an\n   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY\n   KIND, either express or implied.  See the License for the\n   specific language governing permissions and limitations\n   under the License.\n'
'\nKeeps information about historical releases.\n'
ReleaseInfo = namedtuple('ReleaseInfo', 'release_version release_version_no_leading_zeros last_commit_hash content file_name')

def strip_leading_zeros(version: str) -> str:
    if False:
        i = 10
        return i + 15
    '\n    Strips leading zeros from version number.\n\n    This converts 1974.04.03 to 1974.4.3 as the format with leading month and day zeros is not accepted\n    by PIP versioning.\n\n    :param version: version number in CALVER format (potentially with leading 0s in date and month)\n    :return: string with leading 0s after dot replaced.\n    '
    return '.'.join((str(int(i)) for i in version.split('.')))

def get_previous_release_info(previous_release_version: str | None, past_releases: list[ReleaseInfo], current_release_version: str) -> str | None:
    if False:
        i = 10
        return i + 15
    'Find previous release.\n\n    In case we are re-running current release, we assume that last release was\n    the previous one. This is needed so that we can generate list of changes\n    since the previous release.\n\n    :param previous_release_version: known last release version\n    :param past_releases: list of past releases\n    :param current_release_version: release that we are working on currently\n    '
    previous_release = None
    if previous_release_version == current_release_version:
        if len(past_releases) > 1:
            previous_release = past_releases[1].last_commit_hash
    else:
        previous_release = past_releases[0].last_commit_hash if past_releases else None
    return previous_release

def check_if_release_version_ok(past_releases: list[ReleaseInfo], current_release_version: str) -> tuple[str, str | None]:
    if False:
        i = 10
        return i + 15
    'Check if the release version passed is not later than the last release version.\n\n    :param past_releases: all past releases (if there are any)\n    :param current_release_version: release version to check\n    :return: Tuple of current/previous_release (previous might be None if there are no releases)\n    '
    previous_release_version = past_releases[0].release_version if past_releases else None
    if current_release_version == '':
        if previous_release_version:
            current_release_version = previous_release_version
        else:
            current_release_version = (datetime.today() + timedelta(days=5)).strftime('%Y.%m.%d')
    if previous_release_version:
        if Version(current_release_version) < Version(previous_release_version):
            console.print(f'[red]The release {current_release_version} must be not less than {previous_release_version} - last release for the package[/]')
            raise Exception('Bad release version')
    return (current_release_version, previous_release_version)

def get_cross_provider_dependent_packages(provider_package_id: str) -> list[str]:
    if False:
        print('Hello World!')
    'Returns cross-provider dependencies for the package.\n\n    :param provider_package_id: package id\n    :return: list of cross-provider dependencies\n    '
    if provider_package_id in get_removed_provider_ids():
        return []
    return ALL_DEPENDENCIES[provider_package_id][CROSS_PROVIDERS_DEPS]

def make_current_directory_safe(verbose: bool):
    if False:
        i = 10
        return i + 15
    'Makes current directory safe for Git.\n\n    New git checks if git ownership for the folder is not manipulated with. We\n    are running this command only inside the container where the directory is\n    mounted from "regular" user to "root" user which is used inside the\n    container, so this is quite ok to assume the directory it is used is safe.\n\n    It\'s also ok to leave it as safe - it is a global option inside the\n    container so it will disappear when we exit.\n\n    :param verbose: whether to print commands being executed\n    '
    safe_dir_remove_command = ['git', 'config', '--global', '--unset-all', 'safe.directory']
    if verbose:
        console.print(f"Running command: '{' '.join(safe_dir_remove_command)}'")
    subprocess.call(safe_dir_remove_command)
    safe_dir_add_command = ['git', 'config', '--global', '--add', 'safe.directory', '/opt/airflow']
    if verbose:
        console.print(f"Running command: '{' '.join(safe_dir_add_command)}'")
    subprocess.check_call(safe_dir_add_command)

def get_git_tag_check_command(tag: str) -> list[str]:
    if False:
        while True:
            i = 10
    'Get git command to check if tag exits.\n\n    :param tag: Tag to check\n    :return: git command to run\n    '
    return ['git', 'rev-parse', tag]

def get_source_package_path(provider_package_id: str) -> str:
    if False:
        print('Hello World!')
    'Retrieves source package path from package id.\n\n    :param provider_package_id: id of the package\n    :return: path of the providers folder\n    '
    return os.path.join(PROVIDERS_PATH, *provider_package_id.split('.'))

def get_documentation_package_path(provider_package_id: str) -> Path:
    if False:
        return 10
    'Retrieves documentation package path from package id.\n\n    :param provider_package_id: id of the package\n    :return: path of the documentation folder\n    '
    return DOCUMENTATION_PATH / f"apache-airflow-providers-{provider_package_id.replace('.', '-')}"

def get_generated_package_path(provider_package_id: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Retrieves generated package path from package id.\n\n    :param provider_package_id: id of the package\n    :return: path of the providers folder\n    '
    provider_package_path = os.path.join(GENERATED_PROVIDERS_PATH, *provider_package_id.split('.'))
    return provider_package_path

def validate_provider_info_with_runtime_schema(provider_info: dict[str, Any]) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Validates provider info against the runtime schema.\n\n    This way we check if the provider info in the packages is future-compatible.\n    The Runtime Schema should only change when there is a major version change.\n\n    :param provider_info: provider info to validate\n    '
    with open(PROVIDER_RUNTIME_DATA_SCHEMA_PATH) as schema_file:
        schema = json.load(schema_file)
    try:
        jsonschema.validate(provider_info, schema=schema)
    except jsonschema.ValidationError as ex:
        console.print('[red]Provider info not validated against runtime schema[/]')
        raise Exception('Error when validating schema. The schema must be compatible with airflow/provider_info.schema.json.', ex)

def get_provider_yaml(provider_package_id: str) -> dict[str, Any]:
    if False:
        return 10
    'Retrieves provider info from the provider YAML file.\n\n    The provider yaml file contains more information than provider_info that is\n    used at runtime. This method converts the full provider yaml file into\n    stripped-down provider info and validates it against deprecated 2.0.0 schema\n    and runtime schema.\n\n    :param provider_package_id: package id to retrieve provider.yaml from\n    :return: provider_info dictionary\n    '
    provider_yaml_file_name = os.path.join(get_source_package_path(provider_package_id), 'provider.yaml')
    if not os.path.exists(provider_yaml_file_name):
        raise Exception(f'The provider.yaml file is missing: {provider_yaml_file_name}')
    with open(provider_yaml_file_name) as provider_file:
        provider_yaml_dict = safe_load(provider_file)
    return provider_yaml_dict

def get_provider_info_from_provider_yaml(provider_package_id: str) -> dict[str, Any]:
    if False:
        while True:
            i = 10
    'Retrieves provider info from the provider yaml file.\n\n    :param provider_package_id: package id to retrieve provider.yaml from\n    :return: provider_info dictionary\n    '
    provider_yaml_dict = get_provider_yaml(provider_package_id=provider_package_id)
    validate_provider_info_with_runtime_schema(provider_yaml_dict)
    return provider_yaml_dict

def get_version_tag(version: str, provider_package_id: str, version_suffix: str=''):
    if False:
        for i in range(10):
            print('nop')
    if version_suffix is None:
        version_suffix = ''
    return f"providers-{provider_package_id.replace('.', '-')}/{version}{version_suffix}"

def get_provider_details(provider_package_id: str) -> ProviderPackageDetails:
    if False:
        i = 10
        return i + 15
    provider_info = get_provider_info_from_provider_yaml(provider_package_id)
    plugins: list[PluginInfo] = []
    if 'plugins' in provider_info:
        for plugin in provider_info['plugins']:
            (package_name, class_name) = plugin['plugin-class'].rsplit('.', maxsplit=1)
            plugins.append(PluginInfo(name=plugin['name'], package_name=package_name, class_name=class_name))
    return ProviderPackageDetails(provider_package_id=provider_package_id, full_package_name=f'airflow.providers.{provider_package_id}', pypi_package_name=f"apache-airflow-providers-{provider_package_id.replace('.', '-')}", source_provider_package_path=get_source_package_path(provider_package_id), documentation_provider_package_path=get_documentation_package_path(provider_package_id), provider_description=provider_info['description'], versions=provider_info['versions'], excluded_python_versions=provider_info.get('excluded-python-versions') or [], plugins=plugins, removed=provider_info.get('removed', False))

def get_provider_requirements(provider_package_id: str) -> list[str]:
    if False:
        i = 10
        return i + 15
    provider_yaml = get_provider_yaml(provider_package_id)
    return provider_yaml['dependencies']

def get_provider_jinja_context(provider_info: dict[str, Any], provider_details: ProviderPackageDetails, current_release_version: str, version_suffix: str):
    if False:
        for i in range(10):
            print('nop')
    verify_provider_package(provider_details.provider_package_id)
    changelog_path = verify_changelog_exists(provider_details.provider_package_id)
    cross_providers_dependencies = get_cross_provider_dependent_packages(provider_package_id=provider_details.provider_package_id)
    release_version_no_leading_zeros = strip_leading_zeros(current_release_version)
    pip_requirements_table = convert_pip_requirements_to_table(get_provider_requirements(provider_details.provider_package_id))
    pip_requirements_table_rst = convert_pip_requirements_to_table(get_provider_requirements(provider_details.provider_package_id), markdown=False)
    cross_providers_dependencies_table_rst = convert_cross_package_dependencies_to_table(cross_providers_dependencies, markdown=False)
    with open(changelog_path) as changelog_file:
        changelog = changelog_file.read()
    supported_python_versions = [p for p in ALL_PYTHON_VERSIONS if p not in provider_details.excluded_python_versions]
    python_requires = '~=3.8'
    for p in provider_details.excluded_python_versions:
        python_requires += f', !={p}'
    min_airflow_version = MIN_AIRFLOW_VERSION
    for dependency in provider_info['dependencies']:
        if dependency.startswith('apache-airflow>='):
            current_min_airflow_version = dependency.split('>=')[1]
            if Version(current_min_airflow_version) > Version(min_airflow_version):
                min_airflow_version = current_min_airflow_version
    context: dict[str, Any] = {'ENTITY_TYPES': list(EntityType), 'README_FILE': 'README.rst', 'PROVIDER_PACKAGE_ID': provider_details.provider_package_id, 'PACKAGE_PIP_NAME': get_pip_package_name(provider_details.provider_package_id), 'PACKAGE_WHEEL_NAME': get_wheel_package_name(provider_details.provider_package_id), 'FULL_PACKAGE_NAME': provider_details.full_package_name, 'PROVIDER_PATH': provider_details.full_package_name.replace('.', '/'), 'RELEASE': current_release_version, 'RELEASE_NO_LEADING_ZEROS': release_version_no_leading_zeros, 'VERSION_SUFFIX': version_suffix or '', 'CROSS_PROVIDERS_DEPENDENCIES': cross_providers_dependencies, 'PIP_REQUIREMENTS': get_provider_requirements(provider_details.provider_package_id), 'PROVIDER_TYPE': 'Provider', 'PROVIDERS_FOLDER': 'providers', 'PROVIDER_DESCRIPTION': provider_details.provider_description, 'INSTALL_REQUIREMENTS': get_install_requirements(provider_package_id=provider_details.provider_package_id, version_suffix=version_suffix), 'SETUP_REQUIREMENTS': get_setup_requirements(), 'EXTRAS_REQUIREMENTS': get_package_extras(provider_package_id=provider_details.provider_package_id), 'CROSS_PROVIDERS_DEPENDENCIES_TABLE_RST': cross_providers_dependencies_table_rst, 'PIP_REQUIREMENTS_TABLE': pip_requirements_table, 'PIP_REQUIREMENTS_TABLE_RST': pip_requirements_table_rst, 'PROVIDER_INFO': provider_info, 'CHANGELOG_RELATIVE_PATH': os.path.relpath(provider_details.source_provider_package_path, provider_details.documentation_provider_package_path), 'CHANGELOG': changelog, 'SUPPORTED_PYTHON_VERSIONS': supported_python_versions, 'PYTHON_REQUIRES': python_requires, 'PLUGINS': provider_details.plugins, 'MIN_AIRFLOW_VERSION': min_airflow_version, 'PREINSTALLED_PROVIDER': provider_details.provider_package_id in PREINSTALLED_PROVIDERS, 'PROVIDER_REMOVED': provider_details.removed}
    return context

def prepare_readme_file(context):
    if False:
        return 10
    readme_content = LICENCE_RST + render_template(template_name='PROVIDER_README', context=context, extension='.rst')
    readme_file_path = os.path.join(TARGET_PROVIDER_PACKAGES_PATH, 'README.rst')
    with open(readme_file_path, 'w') as readme_file:
        readme_file.write(readme_content)

def update_setup_files(provider_package_id: str, version_suffix: str):
    if False:
        while True:
            i = 10
    'Updates generated setup.cfg/setup.py/manifest.in/provider_info for packages.\n\n    :param provider_package_id: id of the package\n    :param version_suffix: version suffix corresponding to the version in the code\n    :returns False if the package should be skipped, True if everything generated properly\n    '
    verify_provider_package(provider_package_id)
    provider_details = get_provider_details(provider_package_id)
    provider_info = get_provider_info_from_provider_yaml(provider_package_id)
    current_release_version = provider_details.versions[0]
    jinja_context = get_provider_jinja_context(provider_info=provider_info, provider_details=provider_details, current_release_version=current_release_version, version_suffix=version_suffix)
    console.print()
    console.print(f'Generating setup files for {provider_package_id}')
    console.print()
    prepare_setup_py_file(jinja_context)
    prepare_setup_cfg_file(jinja_context)
    prepare_get_provider_info_py_file(jinja_context, provider_package_id)
    prepare_manifest_in_file(jinja_context)
    prepare_readme_file(jinja_context)
    return True

def replace_content(file_path, old_text, new_text, provider_package_id):
    if False:
        i = 10
        return i + 15
    if new_text != old_text:
        (_, temp_file_path) = tempfile.mkstemp()
        try:
            if os.path.isfile(file_path):
                copyfile(file_path, temp_file_path)
            with open(file_path, 'w') as readme_file:
                readme_file.write(new_text)
            console.print()
            console.print(f'Generated {file_path} file for the {provider_package_id} provider')
            console.print()
            if old_text != '':
                subprocess.call(['diff', '--color=always', temp_file_path, file_path])
        finally:
            os.remove(temp_file_path)
AUTOMATICALLY_GENERATED_MARKER = 'AUTOMATICALLY GENERATED'
AUTOMATICALLY_GENERATED_CONTENT = f'.. THE REMAINDER OF THE FILE IS {AUTOMATICALLY_GENERATED_MARKER}. IT WILL BE OVERWRITTEN AT RELEASE TIME!'
BACKTICKS_CHECK = re.compile('^(?!    ).*(^| )`[^`]+`([^_]|$)', re.MULTILINE)

def _update_file(context: dict[str, Any], template_name: str, extension: str, file_name: str, provider_package_id: str, target_path: Path, regenerate_missing_docs: bool) -> bool:
    if False:
        print('Hello World!')
    file_path = target_path / file_name
    if regenerate_missing_docs and file_path.exists():
        return True
    new_text = render_template(template_name=template_name, context=context, extension=extension, keep_trailing_newline=True)
    file_path = target_path / file_name
    old_text = ''
    if os.path.isfile(file_path):
        with open(file_path) as readme_file_read:
            old_text = readme_file_read.read()
    replace_content(file_path, old_text, new_text, provider_package_id)
    index_path = target_path / 'index.rst'
    if not index_path.exists():
        console.print(f'[red]ERROR! The index must exist for the provider docs: {index_path}')
        sys.exit(1)
    expected_link_in_index = f"<{file_name.split('.')[0]}>"
    if expected_link_in_index not in index_path.read_text():
        console.print(f'\n[red]ERROR! The {index_path} must contain link to the generated documentation:[/]\n\n[yellow]{expected_link_in_index}[/]\n\n[bright_blue]Please make sure to add it to {index_path}.\n')
    console.print(f'Checking for backticks correctly generated in: {file_path}')
    match = BACKTICKS_CHECK.search(file_path.read_text())
    if match:
        console.print(f'\n[red]ERROR: Single backticks (`) found in {file_path}:[/]\n\n[yellow]{match.group(0)}[/]\n\n[bright_blue]Please fix them by replacing with double backticks (``).[/]\n')
        return False
    console.print(f'[green]Generated {file_path} for {provider_package_id} is OK[/]')
    return True

@lru_cache(maxsize=None)
def black_mode() -> Mode:
    if False:
        while True:
            i = 10
    config = parse_pyproject_toml(os.path.join(AIRFLOW_SOURCES_ROOT_PATH, 'pyproject.toml'))
    target_versions = {TargetVersion[val.upper()] for val in config.get('target_version', ())}
    return Mode(target_versions=target_versions, line_length=config.get('line_length', Mode.line_length))

def black_format(content) -> str:
    if False:
        for i in range(10):
            print('nop')
    return format_str(content, mode=black_mode())

def prepare_setup_py_file(context):
    if False:
        for i in range(10):
            print('nop')
    setup_py_template_name = 'SETUP'
    setup_py_file_path = os.path.abspath(os.path.join(get_target_folder(), 'setup.py'))
    setup_py_content = render_template(template_name=setup_py_template_name, context=context, extension='.py', autoescape=False)
    with open(setup_py_file_path, 'w') as setup_py_file:
        setup_py_file.write(black_format(setup_py_content))

def prepare_setup_cfg_file(context):
    if False:
        print('Hello World!')
    setup_cfg_template_name = 'SETUP'
    setup_cfg_file_path = os.path.abspath(os.path.join(get_target_folder(), 'setup.cfg'))
    setup_cfg_content = render_template(template_name=setup_cfg_template_name, context=context, extension='.cfg', autoescape=False, keep_trailing_newline=True)
    with open(setup_cfg_file_path, 'w') as setup_cfg_file:
        setup_cfg_file.write(setup_cfg_content)

def prepare_get_provider_info_py_file(context, provider_package_id: str):
    if False:
        print('Hello World!')
    get_provider_template_name = 'get_provider_info'
    get_provider_file_path = os.path.abspath(os.path.join(get_target_providers_package_folder(provider_package_id), 'get_provider_info.py'))
    get_provider_content = render_template(template_name=get_provider_template_name, context=context, extension='.py', autoescape=False, keep_trailing_newline=True)
    with open(get_provider_file_path, 'w') as get_provider_file:
        get_provider_file.write(black_format(get_provider_content))

def prepare_manifest_in_file(context):
    if False:
        print('Hello World!')
    target = os.path.abspath(os.path.join(get_target_folder(), 'MANIFEST.in'))
    content = render_template(template_name='MANIFEST', context=context, extension='.in', autoescape=False, keep_trailing_newline=True)
    with open(target, 'w') as fh:
        fh.write(content)

def get_all_providers() -> list[str]:
    if False:
        while True:
            i = 10
    'Returns all providers for regular packages.\n\n    :return: list of providers that are considered for provider packages\n    '
    return list(ALL_PROVIDERS)

def get_removed_provider_ids() -> list[str]:
    if False:
        return 10
    '\n    Yields the ids of suspended providers.\n    '
    import yaml
    removed_provider_ids = []
    for provider_path in PROVIDERS_PATH.rglob('provider.yaml'):
        provider_yaml = yaml.safe_load(provider_path.read_text())
        package_name = provider_yaml.get('package-name')
        if provider_yaml.get('removed', False):
            if not provider_yaml.get('suspended'):
                console.print(f'[error]The provider {package_name} is marked for removal in provider.yaml, but not suspended. Please suspend the provider first before removing it.\n')
                sys.exit(1)
            removed_provider_ids.append(package_name[len('apache-airflow-providers-'):].replace('-', '.'))
    return removed_provider_ids

def verify_provider_package(provider_package_id: str) -> None:
    if False:
        return 10
    'Verifies if the provider package is good.\n\n    :param provider_package_id: package id to verify\n    '
    if provider_package_id not in get_all_providers():
        if provider_package_id in get_removed_provider_ids():
            console.print()
            console.print(f'[yellow]The package: {provider_package_id} is suspended, but since you asked for it, it will be built [/]')
            console.print()
        else:
            console.print(f'[red]Wrong package name: {provider_package_id}[/]')
            console.print('Use one of:')
            console.print(get_all_providers())
            console.print(f'[red]The package {provider_package_id} is not a provider package.')
            sys.exit(1)

def verify_changelog_exists(package: str) -> str:
    if False:
        return 10
    provider_details = get_provider_details(package)
    changelog_path = os.path.join(provider_details.source_provider_package_path, 'CHANGELOG.rst')
    if not os.path.isfile(changelog_path):
        console.print(f'\n[red]ERROR: Missing {changelog_path}[/]\n')
        console.print('[info]Please add the file with initial content:')
        console.print('----- START COPYING AFTER THIS LINE ------- ')
        processed_changelog = jinja2.Template(INITIAL_CHANGELOG_CONTENT, autoescape=True).render(package_name=provider_details.pypi_package_name)
        syntax = Syntax(processed_changelog, 'rst', theme='ansi_dark')
        console.print(syntax)
        console.print('----- END COPYING BEFORE THIS LINE ------- ')
        sys.exit(1)
    return changelog_path

@cli.command()
def list_providers_packages():
    if False:
        for i in range(10):
            print('nop')
    'List all provider packages.'
    providers = get_all_providers()
    providers_to_remove_from_release = []
    for provider in providers:
        if provider not in providers_to_remove_from_release:
            console.print(provider)

def tag_exists_for_version(provider_package_id: str, current_tag: str, verbose: bool):
    if False:
        while True:
            i = 10
    provider_details = get_provider_details(provider_package_id)
    if verbose:
        console.print(f'Checking if tag `{current_tag}` exists.')
    if not subprocess.call(get_git_tag_check_command(current_tag), cwd=provider_details.source_provider_package_path, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL):
        if verbose:
            console.print(f'Tag `{current_tag}` exists.')
        return True
    if verbose:
        console.print(f'Tag `{current_tag}` does not exist.')
    return False

@cli.command()
@option_version_suffix
@option_git_update
@argument_package_id
@option_verbose
@option_skip_tag_check
def generate_setup_files(version_suffix: str, git_update: bool, package_id: str, verbose: bool, skip_tag_check: bool):
    if False:
        for i in range(10):
            print('nop')
    'Generates setup files for the package.\n\n    See `list-providers-packages` subcommand for the possible PACKAGE_ID values.\n    '
    provider_package_id = package_id
    with with_group(f"Generate setup files for '{provider_package_id}'"):
        if not skip_tag_check:
            current_tag = get_current_tag(provider_package_id, version_suffix, git_update, verbose)
            if tag_exists_for_version(provider_package_id, current_tag, verbose):
                console.print(f'[yellow]The tag {current_tag} exists. Not preparing the package.[/]')
                sys.exit(64)
        if update_setup_files(provider_package_id, version_suffix):
            console.print(f'[green]Generated regular package setup files for {provider_package_id}[/]')
        else:
            sys.exit(64)

def get_current_tag(provider_package_id: str, suffix: str, git_update: bool, verbose: bool):
    if False:
        i = 10
        return i + 15
    verify_provider_package(provider_package_id)
    provider_info = get_provider_info_from_provider_yaml(provider_package_id)
    versions: list[str] = provider_info['versions']
    current_version = versions[0]
    current_tag = get_version_tag(current_version, provider_package_id, suffix)
    return current_tag

def cleanup_remnants(verbose: bool):
    if False:
        while True:
            i = 10
    if verbose:
        console.print('Cleaning remnants')
    files = glob.glob('*.egg-info')
    for file in files:
        shutil.rmtree(file, ignore_errors=True)
    files = glob.glob('build')
    for file in files:
        shutil.rmtree(file, ignore_errors=True)

def verify_setup_cfg_prepared(provider_package):
    if False:
        for i in range(10):
            print('nop')
    with open('setup.cfg') as f:
        setup_content = f.read()
    search_for = f"providers-{provider_package.replace('.', '-')} for Apache Airflow"
    if search_for not in setup_content:
        console.print(f'[red]The setup.py is probably prepared for another package. It does not contain [bold]{search_for}[/bold]![/]')
        console.print(f'\nRun:\n\n[bold]./dev/provider_packages/prepare_provider_packages.py generate-setup-files {provider_package}[/bold]\n')
        raise Exception('Wrong setup!')

@cli.command()
@option_package_format
@option_git_update
@option_version_suffix
@argument_package_id
@option_verbose
@option_skip_tag_check
def build_provider_packages(package_format: str, git_update: bool, version_suffix: str, package_id: str, verbose: bool, skip_tag_check: bool):
    if False:
        for i in range(10):
            print('nop')
    'Builds provider package.\n\n    See `list-providers-packages` subcommand for the possible PACKAGE_ID values.\n    '
    import tempfile
    tmp_build_dir = tempfile.TemporaryDirectory().name
    tmp_dist_dir = tempfile.TemporaryDirectory().name
    try:
        provider_package_id = package_id
        with with_group(f"Prepare provider package for '{provider_package_id}'"):
            if not skip_tag_check and (version_suffix.startswith('rc') or version_suffix == ''):
                released_tag = get_current_tag(provider_package_id, '', git_update, verbose)
                if tag_exists_for_version(provider_package_id, released_tag, verbose):
                    console.print(f'[yellow]The tag {released_tag} exists. Skipping the package.[/]')
                    return False
            console.print(f'Changing directory to {TARGET_PROVIDER_PACKAGES_PATH}')
            os.chdir(TARGET_PROVIDER_PACKAGES_PATH)
            cleanup_remnants(verbose)
            provider_package = package_id
            verify_setup_cfg_prepared(provider_package)
            console.print(f'Building provider package: {provider_package} in format {package_format}')
            command: list[str] = ['python3', 'setup.py', 'build', '--build-temp', tmp_build_dir]
            if version_suffix is not None:
                command.extend(['egg_info', '--tag-build', version_suffix])
            if package_format in ['sdist', 'both']:
                command.append('sdist')
            if package_format in ['wheel', 'both']:
                command.extend(['bdist_wheel', '--bdist-dir', tmp_dist_dir])
            console.print(f"Executing command: '{' '.join(command)}'")
            try:
                subprocess.check_call(args=command, stdout=subprocess.DEVNULL)
            except subprocess.CalledProcessError as ex:
                console.print('[red]The command returned an error %s', ex)
                sys.exit(ex.returncode)
            console.print(f'[green]Prepared provider package {provider_package} in format {package_format}[/]')
    finally:
        shutil.rmtree(tmp_build_dir, ignore_errors=True)
        shutil.rmtree(tmp_dist_dir, ignore_errors=True)
if __name__ == '__main__':
    try:
        cli()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(65)
        except SystemExit:
            os._exit(65)