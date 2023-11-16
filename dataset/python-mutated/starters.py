"""kedro is a CLI for managing Kedro projects.

This module implements commands available from the kedro CLI for creating
projects.
"""
from __future__ import annotations
import os
import re
import shutil
import stat
import tempfile
import warnings
from collections import OrderedDict
from itertools import groupby
from pathlib import Path
from typing import Any, Callable
import click
import yaml
from attrs import define, field
import kedro
from kedro import KedroDeprecationWarning
from kedro import __version__ as version
from kedro.framework.cli.utils import CONTEXT_SETTINGS, KedroCliError, _clean_pycache, _get_entry_points, _safe_load_entry_point, command_with_verbosity
CONFIG_ARG_HELP = "Non-interactive mode, using a configuration yaml file. This file\nmust supply  the keys required by the template's prompts.yml. When not using a starter,\nthese are `project_name`, `repo_name` and `python_package`."
STARTER_ARG_HELP = 'Specify the starter template to use when creating the project.\nThis can be the path to a local directory, a URL to a remote VCS repository supported\nby `cookiecutter` or one of the aliases listed in ``kedro starter list``.\n'
CHECKOUT_ARG_HELP = 'An optional tag, branch or commit to checkout in the starter repository.'
DIRECTORY_ARG_HELP = 'An optional directory inside the repository where the starter resides.'

@define(order=True)
class KedroStarterSpec:
    """Specification of custom kedro starter template
    Args:
        alias: alias of the starter which shows up on `kedro starter list` and is used
        by the starter argument of `kedro new`
        template_path: path to a directory or a URL to a remote VCS repository supported
        by `cookiecutter`
        directory: optional directory inside the repository where the starter resides.
        origin: reserved field used by kedro internally to determine where the starter
        comes from, users do not need to provide this field.
    """
    alias: str
    template_path: str
    directory: str | None = None
    origin: str | None = field(init=False)
KEDRO_PATH = Path(kedro.__file__).parent
TEMPLATE_PATH = KEDRO_PATH / 'templates' / 'project'
_DEPRECATED_STARTERS = ['pandas-iris', 'pyspark-iris', 'pyspark', 'standalone-datacatalog']
_STARTERS_REPO = 'git+https://github.com/kedro-org/kedro-starters.git'
_OFFICIAL_STARTER_SPECS = [KedroStarterSpec('astro-airflow-iris', _STARTERS_REPO, 'astro-airflow-iris'), KedroStarterSpec('astro-iris', _STARTERS_REPO, 'astro-airflow-iris'), KedroStarterSpec('standalone-datacatalog', _STARTERS_REPO, 'standalone-datacatalog'), KedroStarterSpec('pandas-iris', _STARTERS_REPO, 'pandas-iris'), KedroStarterSpec('pyspark', _STARTERS_REPO, 'pyspark'), KedroStarterSpec('pyspark-iris', _STARTERS_REPO, 'pyspark-iris'), KedroStarterSpec('spaceflights', _STARTERS_REPO, 'spaceflights'), KedroStarterSpec('databricks-iris', _STARTERS_REPO, 'databricks-iris')]
for starter_spec in _OFFICIAL_STARTER_SPECS:
    starter_spec.origin = 'kedro'
_OFFICIAL_STARTER_SPECS = {spec.alias: spec for spec in _OFFICIAL_STARTER_SPECS}

@click.group(context_settings=CONTEXT_SETTINGS, name='Kedro')
def create_cli():
    if False:
        print('Hello World!')
    pass

@create_cli.group()
def starter():
    if False:
        for i in range(10):
            print('nop')
    'Commands for working with project starters.'

@command_with_verbosity(create_cli, short_help='Create a new kedro project.')
@click.option('--config', '-c', 'config_path', type=click.Path(exists=True), help=CONFIG_ARG_HELP)
@click.option('--starter', '-s', 'starter_alias', help=STARTER_ARG_HELP)
@click.option('--checkout', help=CHECKOUT_ARG_HELP)
@click.option('--directory', help=DIRECTORY_ARG_HELP)
def new(config_path, starter_alias, checkout, directory, **kwargs):
    if False:
        i = 10
        return i + 15
    'Create a new kedro project.'
    if starter_alias in _DEPRECATED_STARTERS:
        warnings.warn(f"The starter '{starter_alias}' has been deprecated and will be archived from Kedro 0.19.0.", KedroDeprecationWarning)
    click.secho('From Kedro 0.19.0, the command `kedro new` will come with the option of interactively selecting add-ons for your project such as linting, testing, custom logging, and more. The selected add-ons will add the basic setup for the utilities selected to your projects.', fg='green')
    if checkout and (not starter_alias):
        raise KedroCliError('Cannot use the --checkout flag without a --starter value.')
    if directory and (not starter_alias):
        raise KedroCliError('Cannot use the --directory flag without a --starter value.')
    starters_dict = _get_starters_dict()
    if starter_alias in starters_dict:
        if directory:
            raise KedroCliError('Cannot use the --directory flag with a --starter alias.')
        spec = starters_dict[starter_alias]
        template_path = spec.template_path
        directory = spec.directory
        checkout = checkout or version
    elif starter_alias is not None:
        template_path = starter_alias
        checkout = checkout or version
    else:
        template_path = str(TEMPLATE_PATH)
    tmpdir = tempfile.mkdtemp()
    cookiecutter_dir = _get_cookiecutter_dir(template_path, checkout, directory, tmpdir)
    prompts_required = _get_prompts_required(cookiecutter_dir)
    if not config_path:
        cookiecutter_context = _make_cookiecutter_context_for_prompts(cookiecutter_dir)
    shutil.rmtree(tmpdir, onerror=_remove_readonly)
    if not prompts_required:
        config = {}
        if config_path:
            config = _fetch_config_from_file(config_path)
    elif config_path:
        config = _fetch_config_from_file(config_path)
        _validate_config_file(config, prompts_required)
    else:
        config = _fetch_config_from_user_prompts(prompts_required, cookiecutter_context)
    cookiecutter_args = _make_cookiecutter_args(config, checkout, directory)
    _create_project(template_path, cookiecutter_args)

@starter.command('list')
def list_starters():
    if False:
        return 10
    'List all official project starters available.'
    starters_dict = _get_starters_dict()
    sorted_starters_dict: dict[str, dict[str, KedroStarterSpec]] = {origin: dict(sorted(starters_dict_by_origin)) for (origin, starters_dict_by_origin) in groupby(starters_dict.items(), lambda item: item[1].origin)}
    sorted_starters_dict = dict(sorted(sorted_starters_dict.items(), key=lambda x: x == 'kedro'))
    warnings.warn(f'The starters {_DEPRECATED_STARTERS} are deprecated and will be archived in Kedro 0.19.0.')
    for (origin, starters_spec) in sorted_starters_dict.items():
        click.secho(f'\nStarters from {origin}\n', fg='yellow')
        click.echo(yaml.safe_dump(_starter_spec_to_dict(starters_spec), sort_keys=False))

def _get_cookiecutter_dir(template_path: str, checkout: str, directory: str, tmpdir: str) -> Path:
    if False:
        while True:
            i = 10
    'Gives a path to the cookiecutter directory. If template_path is a repo then\n    clones it to ``tmpdir``; if template_path is a file path then directly uses that\n    path without copying anything.\n    '
    from cookiecutter.exceptions import RepositoryCloneFailed, RepositoryNotFound
    from cookiecutter.repository import determine_repo_dir
    try:
        (cookiecutter_dir, _) = determine_repo_dir(template=template_path, abbreviations={}, clone_to_dir=Path(tmpdir).resolve(), checkout=checkout, no_input=True, directory=directory)
    except (RepositoryNotFound, RepositoryCloneFailed) as exc:
        error_message = f'Kedro project template not found at {template_path}.'
        if checkout:
            error_message += f' Specified tag {checkout}. The following tags are available: ' + ', '.join(_get_available_tags(template_path))
        official_starters = sorted(_OFFICIAL_STARTER_SPECS)
        raise KedroCliError(f'{error_message}. The aliases for the official Kedro starters are: \n{yaml.safe_dump(official_starters, sort_keys=False)}') from exc
    return Path(cookiecutter_dir)

def _get_prompts_required(cookiecutter_dir: Path) -> dict[str, Any] | None:
    if False:
        print('Hello World!')
    'Finds the information a user must supply according to prompts.yml.'
    prompts_yml = cookiecutter_dir / 'prompts.yml'
    if not prompts_yml.is_file():
        return None
    try:
        with prompts_yml.open('r') as prompts_file:
            return yaml.safe_load(prompts_file)
    except Exception as exc:
        raise KedroCliError('Failed to generate project: could not load prompts.yml.') from exc

def _get_available_tags(template_path: str) -> list:
    if False:
        return 10
    import git
    try:
        tags = git.cmd.Git().ls_remote('--tags', template_path.replace('git+', ''))
        unique_tags = {tag.split('/')[-1].replace('^{}', '') for tag in tags.split('\n')}
    except git.GitCommandError:
        return []
    return sorted(unique_tags)

def _get_starters_dict() -> dict[str, KedroStarterSpec]:
    if False:
        while True:
            i = 10
    'This function lists all the starter aliases declared in\n    the core repo and in plugins entry points.\n\n    For example, the output for official kedro starters looks like:\n    {"astro-airflow-iris":\n        KedroStarterSpec(\n            name="astro-airflow-iris",\n            template_path="git+https://github.com/kedro-org/kedro-starters.git",\n            directory="astro-airflow-iris",\n            origin="kedro"\n        ),\n    "astro-iris":\n        KedroStarterSpec(\n            name="astro-iris",\n            template_path="git+https://github.com/kedro-org/kedro-starters.git",\n            directory="astro-airflow-iris",\n            origin="kedro"\n        ),\n    }\n    '
    starter_specs = _OFFICIAL_STARTER_SPECS
    for starter_entry_point in _get_entry_points(name='starters'):
        origin = starter_entry_point.module.split('.')[0]
        specs = _safe_load_entry_point(starter_entry_point) or []
        for spec in specs:
            if not isinstance(spec, KedroStarterSpec):
                click.secho(f"The starter configuration loaded from module {origin}should be a 'KedroStarterSpec', got '{type(spec)}' instead", fg='red')
            elif spec.alias in starter_specs:
                click.secho(f'Starter alias `{spec.alias}` from `{origin}` has been ignored as it is already defined by`{starter_specs[spec.alias].origin}`', fg='red')
            else:
                spec.origin = origin
                starter_specs[spec.alias] = spec
    return starter_specs

def _fetch_config_from_file(config_path: str) -> dict[str, str]:
    if False:
        i = 10
        return i + 15
    'Obtains configuration for a new kedro project non-interactively from a file.\n\n    Args:\n        config_path: The path of the config.yml which should contain the data required\n            by ``prompts.yml``.\n\n    Returns:\n        Configuration for starting a new project. This is passed as ``extra_context``\n            to cookiecutter and will overwrite the cookiecutter.json defaults.\n\n    Raises:\n        KedroCliError: If the file cannot be parsed.\n\n    '
    try:
        with open(config_path, encoding='utf-8') as config_file:
            config = yaml.safe_load(config_file)
        if KedroCliError.VERBOSE_ERROR:
            click.echo(config_path + ':')
            click.echo(yaml.dump(config, default_flow_style=False))
    except Exception as exc:
        raise KedroCliError(f'Failed to generate project: could not load config at {config_path}.') from exc
    return config

def _fetch_config_from_user_prompts(prompts: dict[str, Any], cookiecutter_context: OrderedDict) -> dict[str, str]:
    if False:
        i = 10
        return i + 15
    'Interactively obtains information from user prompts.\n\n    Args:\n        prompts: Prompts from prompts.yml.\n        cookiecutter_context: Cookiecutter context generated from cookiecutter.json.\n\n    Returns:\n        Configuration for starting a new project. This is passed as ``extra_context``\n            to cookiecutter and will overwrite the cookiecutter.json defaults.\n    '
    from cookiecutter.environment import StrictEnvironment
    from cookiecutter.prompt import read_user_variable, render_variable
    config: dict[str, str] = {}
    for (variable_name, prompt_dict) in prompts.items():
        prompt = _Prompt(**prompt_dict)
        cookiecutter_variable = render_variable(env=StrictEnvironment(context=cookiecutter_context), raw=cookiecutter_context.get(variable_name), cookiecutter_dict=config)
        user_input = read_user_variable(str(prompt), cookiecutter_variable)
        if user_input:
            prompt.validate(user_input)
            config[variable_name] = user_input
    return config

def _make_cookiecutter_context_for_prompts(cookiecutter_dir: Path):
    if False:
        i = 10
        return i + 15
    from cookiecutter.generate import generate_context
    cookiecutter_context = generate_context(cookiecutter_dir / 'cookiecutter.json')
    return cookiecutter_context.get('cookiecutter', {})

def _make_cookiecutter_args(config: dict[str, str], checkout: str, directory: str) -> dict[str, Any]:
    if False:
        i = 10
        return i + 15
    "Creates a dictionary of arguments to pass to cookiecutter.\n\n    Args:\n        config: Configuration for starting a new project. This is passed as\n            ``extra_context`` to cookiecutter and will overwrite the cookiecutter.json\n            defaults.\n        checkout: The tag, branch or commit in the starter repository to checkout.\n            Maps directly to cookiecutter's ``checkout`` argument. Relevant only when\n            using a starter.\n        directory: The directory of a specific starter inside a repository containing\n            multiple starters. Maps directly to cookiecutter's ``directory`` argument.\n            Relevant only when using a starter.\n            https://cookiecutter.readthedocs.io/en/1.7.2/advanced/directories.html\n\n    Returns:\n        Arguments to pass to cookiecutter.\n    "
    config.setdefault('kedro_version', version)
    cookiecutter_args = {'output_dir': config.get('output_dir', str(Path.cwd().resolve())), 'no_input': True, 'extra_context': config}
    if checkout:
        cookiecutter_args['checkout'] = checkout
    if directory:
        cookiecutter_args['directory'] = directory
    return cookiecutter_args

def _validate_config_file(config: dict[str, str], prompts: dict[str, Any]):
    if False:
        return 10
    'Checks that the configuration file contains all needed variables.\n\n    Args:\n        config: The config as a dictionary.\n        prompts: Prompts from prompts.yml.\n\n    Raises:\n        KedroCliError: If the config file is empty or does not contain all the keys\n            required in prompts, or if the output_dir specified does not exist.\n    '
    if config is None:
        raise KedroCliError('Config file is empty.')
    missing_keys = set(prompts) - set(config)
    if missing_keys:
        click.echo(yaml.dump(config, default_flow_style=False))
        raise KedroCliError(f"{', '.join(missing_keys)} not found in config file.")
    if 'output_dir' in config and (not Path(config['output_dir']).exists()):
        raise KedroCliError(f"'{config['output_dir']}' is not a valid output directory. It must be a relative or absolute path to an existing directory.")

def _create_project(template_path: str, cookiecutter_args: dict[str, Any]):
    if False:
        return 10
    'Creates a new kedro project using cookiecutter.\n\n    Args:\n        template_path: The path to the cookiecutter template to create the project.\n            It could either be a local directory or a remote VCS repository\n            supported by cookiecutter. For more details, please see:\n            https://cookiecutter.readthedocs.io/en/latest/usage.html#generate-your-project\n        cookiecutter_args: Arguments to pass to cookiecutter.\n\n    Raises:\n        KedroCliError: If it fails to generate a project.\n    '
    from cookiecutter.main import cookiecutter
    try:
        result_path = cookiecutter(template=template_path, **cookiecutter_args)
    except Exception as exc:
        raise KedroCliError('Failed to generate project when running cookiecutter.') from exc
    _clean_pycache(Path(result_path))
    extra_context = cookiecutter_args['extra_context']
    project_name = extra_context.get('project_name', 'New Kedro Project')
    python_package = extra_context.get('python_package', project_name.lower().replace(' ', '_').replace('-', '_'))
    click.secho(f"\nThe project name '{project_name}' has been applied to: \n- The project title in {result_path}/README.md \n- The folder created for your project in {result_path} \n- The project's python package in {result_path}/src/{python_package}")
    click.secho("\nA best-practice setup includes initialising git and creating a virtual environment before running 'pip install -r src/requirements.txt' to install project-specific dependencies. Refer to the Kedro documentation: https://kedro.readthedocs.io/")
    click.secho(f"\nChange directory to the project generated in {result_path} by entering 'cd {result_path}'", fg='green')

class _Prompt:
    """Represent a single CLI prompt for `kedro new`"""

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        try:
            self.title = kwargs['title']
        except KeyError as exc:
            raise KedroCliError('Each prompt must have a title field to be valid.') from exc
        self.text = kwargs.get('text', '')
        self.regexp = kwargs.get('regex_validator', None)
        self.error_message = kwargs.get('error_message', '')

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        title = self.title.strip().title()
        title = click.style(title + '\n' + '=' * len(title), bold=True)
        prompt_lines = [title] + [self.text]
        prompt_text = '\n'.join((str(line).strip() for line in prompt_lines))
        return f'\n{prompt_text}\n'

    def validate(self, user_input: str) -> None:
        if False:
            return 10
        'Validate a given prompt value against the regex validator'
        if self.regexp and (not re.match(self.regexp, user_input)):
            message = f"'{user_input}' is an invalid value for {self.title}."
            click.secho(message, fg='red', err=True)
            click.secho(self.error_message, fg='red', err=True)
            raise ValueError(message, self.error_message)

def _remove_readonly(func: Callable, path: Path, excinfo: tuple):
    if False:
        while True:
            i = 10
    'Remove readonly files on Windows\n    See: https://docs.python.org/3/library/shutil.html?highlight=shutil#rmtree-example\n    '
    os.chmod(path, stat.S_IWRITE)
    func(path)

def _starter_spec_to_dict(starter_specs: dict[str, KedroStarterSpec]) -> dict[str, dict[str, str]]:
    if False:
        while True:
            i = 10
    'Convert a dictionary of starters spec to a nicely formatted dictionary'
    format_dict: dict[str, dict[str, str]] = {}
    for (alias, spec) in starter_specs.items():
        if alias in _DEPRECATED_STARTERS:
            key = alias + ' (deprecated)'
        else:
            key = alias
        format_dict[key] = {}
        format_dict[key]['template_path'] = spec.template_path
        if spec.directory:
            format_dict[key]['directory'] = spec.directory
    return format_dict