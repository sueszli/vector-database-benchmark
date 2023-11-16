import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional
import click
from jinja2 import Environment, FileSystemLoader
from rich import print
from rich.panel import Panel
from rich.status import Status
from rich.text import Text
from rich.tree import Tree
import lightning.app
_REPORT_HELP_TEXTS = {'core': 'Important files for the app such as various components', 'source': 'A copy of all your source code, including the PL script ⚡', 'tests': 'This app comes with tests!', 'ui': 'Source and build files for the user interface', 'app.py': 'This is the main app file!', 'requirements.txt': 'Lists the dependencies required to be installed before running the app'}
_REPORT_IGNORE_PATTERNS = ['__pycache__', '__init__\\.py', '.*egg-info', '\\..*']

def pl_app(source_dir: str, script_path: str, name: str, overwrite: bool) -> None:
    if False:
        i = 10
        return i + 15
    source_dir = Path(source_dir).resolve()
    script_path = Path(script_path).resolve()
    if not source_dir.is_dir():
        click.echo(f'The given source directory does not exist: {source_dir}', err=True)
        raise SystemExit(1)
    if not script_path.exists():
        click.echo(f'The given script path does not exist: {script_path}', err=True)
        raise SystemExit(1)
    if not script_path.is_file():
        click.echo(f'The given script path must be a file, you passed: {script_path}', err=True)
        raise SystemExit(1)
    if source_dir not in script_path.parents:
        click.echo('The given script path must be a subpath of the source directory. Example: lightning init pl-app ./code ./code/scripts/train.py', err=True)
        raise SystemExit(1)
    rel_script_path = script_path.relative_to(source_dir)
    cwd = Path.cwd()
    destination = cwd / name
    if destination.exists():
        if not overwrite:
            click.echo(f'There is already an app with the name {name} in the current working directory. Choose a different name with `--name` or force to overwrite the existing folder by passing `--overwrite`.', err=True)
            raise SystemExit(1)
        shutil.rmtree(destination)
    template_dir = Path(lightning.app.cli.__file__).parent / 'pl-app-template'
    with Status('[bold green]Copying app files'):
        shutil.copytree(template_dir, destination, ignore=shutil.ignore_patterns('node_modules', 'build'))
        if (template_dir / 'ui' / 'build').exists():
            shutil.copytree(template_dir / 'ui' / 'build', destination / 'ui' / 'build')
        else:
            download_frontend(destination / 'ui' / 'build')
    with Status('[bold green]Copying source files'):
        shutil.copytree(source_dir, destination / 'source', ignore=shutil.ignore_patterns(name))
        project_file_from_template(template_dir, destination, 'app.py', script_path=str(rel_script_path))
        project_file_from_template(template_dir, destination, 'setup.py', app_name=name)
    with Status('[bold green]Installing'):
        subprocess.call(['pip', 'install', '--quiet', '-e', str(destination)])
    print_pretty_report(destination, ignore_patterns=_REPORT_IGNORE_PATTERNS, help_texts=_REPORT_HELP_TEXTS)

def download_frontend(destination: Path) -> None:
    if False:
        i = 10
        return i + 15
    url = 'https://storage.googleapis.com/grid-packages/pytorch-lightning-app/v0.0.0/build.tar.gz'
    build_dir_name = 'build'
    with TemporaryDirectory() as download_dir:
        response = urllib.request.urlopen(url)
        file = tarfile.open(fileobj=response, mode='r|gz')
        file.extractall(path=download_dir)
        shutil.move(str(Path(download_dir, build_dir_name)), destination)

def project_file_from_template(template_dir: Path, destination_dir: Path, template_name: str, **kwargs: Any) -> None:
    if False:
        i = 10
        return i + 15
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_name)
    rendered_template = template.render(**kwargs)
    with open(destination_dir / template_name, 'w') as file:
        file.write(rendered_template)

def print_pretty_report(directory: pathlib.Path, ignore_patterns: Optional[List[str]]=None, help_texts: Optional[Dict[str, str]]=None) -> None:
    if False:
        print('Hello World!')
    'Prints a report for the generated app.'
    tree = Tree(f':open_file_folder: [link file://{directory}]{directory}', guide_style='bold bright_blue')
    help_texts = {} if help_texts is None else help_texts
    paths = sorted(directory.glob('*'), key=lambda p: (p.is_file(), p.name.lower()))
    max_witdth = max((len(p.name) for p in paths))
    patterns_to_ignore = [] if ignore_patterns is None else ignore_patterns
    for path in paths:
        if any((re.match(pattern, path.name) for pattern in patterns_to_ignore)):
            continue
        help_text = help_texts.get(path.name, '')
        padding = ' ' * (max_witdth - len(path.name))
        text_pathname = Text(path.name, 'green')
        text_pathname.highlight_regex('\\..*$', 'bold red')
        text_pathname.stylize(f'link file://{path}')
        text_pathname.append(f' {padding} {help_text}', 'blue')
        icon = '📂 ' if path.is_dir() else '📄 '
        icon = icon if _can_encode_icon(icon) else ''
        tree.add(Text(icon) + text_pathname)
    print('\n')
    print('Done. The app is ready here:\n')
    print(tree)
    print('\nRun it:\n')
    print(Panel(f"[red]lightning run app {directory.relative_to(Path.cwd()) / 'app.py'}"))

def _can_encode_icon(icon: str) -> bool:
    if False:
        i = 10
        return i + 15
    'Helper function to check whether an icon can be encoded.'
    try:
        icon.encode(sys.stdout.encoding)
        return True
    except UnicodeEncodeError:
        return False