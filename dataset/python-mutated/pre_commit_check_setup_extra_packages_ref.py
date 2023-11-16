"""
Checks if all the libraries in setup.py are listed in installation.rst file
"""
from __future__ import annotations
import os
import re
import sys
from pathlib import Path
from rich import print
from rich.console import Console
from rich.table import Table
AIRFLOW_SOURCES_DIR = Path(__file__).parents[3].resolve()
SETUP_PY_FILE = 'setup.py'
DOCS_FILE = os.path.join('docs', 'apache-airflow', 'extra-packages-ref.rst')
PY_IDENTIFIER = '[a-zA-Z_][a-zA-Z0-9_\\.]*'
sys.path.insert(0, os.fspath(AIRFLOW_SOURCES_DIR))
os.environ['_SKIP_PYTHON_VERSION_CHECK'] = 'true'
from setup import add_all_provider_packages, EXTRAS_DEPRECATED_ALIASES, EXTRAS_DEPENDENCIES, PREINSTALLED_PROVIDERS, EXTRAS_DEPRECATED_ALIASES_IGNORED_FROM_REF_DOCS

def get_file_content(*path_elements: str) -> str:
    if False:
        i = 10
        return i + 15
    file_path = AIRFLOW_SOURCES_DIR.joinpath(*path_elements)
    return file_path.read_text()

def get_extras_from_setup() -> set[str]:
    if False:
        i = 10
        return i + 15
    'Returns a set of regular (non-deprecated) extras from setup.'
    return set(EXTRAS_DEPENDENCIES.keys()) - set(EXTRAS_DEPRECATED_ALIASES.keys()) - set(EXTRAS_DEPRECATED_ALIASES_IGNORED_FROM_REF_DOCS)

def get_extras_from_docs() -> set[str]:
    if False:
        return 10
    '\n    Returns a list of extras from airflow.docs.\n    '
    docs_content = get_file_content(DOCS_FILE)
    extras_section_regex = re.compile(f'\\|[^|]+\\|.*pip install .apache-airflow\\[({PY_IDENTIFIER})][^|]+\\|[^|]+\\|', re.MULTILINE)
    doc_extra_set: set[str] = set()
    for doc_extra in extras_section_regex.findall(docs_content):
        doc_extra_set.add(doc_extra)
    return doc_extra_set

def get_preinstalled_providers_from_docs() -> list[str]:
    if False:
        i = 10
        return i + 15
    '\n    Returns list of pre-installed providers from the doc.\n    '
    docs_content = get_file_content(DOCS_FILE)
    preinstalled_section_regex = re.compile(f'\\|\\s*({PY_IDENTIFIER})\\s*\\|[^|]+pip install[^|]+\\|[^|]+\\|\\s+\\*\\s+\\|$', re.MULTILINE)
    return preinstalled_section_regex.findall(docs_content)

def get_deprecated_extras_from_docs() -> dict[str, str]:
    if False:
        i = 10
        return i + 15
    '\n    Returns dict of deprecated extras from airflow.docs (alias -> target extra)\n    '
    deprecated_extras = {}
    docs_content = get_file_content(DOCS_FILE)
    deprecated_extras_section_regex = re.compile('\\| Deprecated extra    \\| Extra to be used instead    \\|\\n(.*)\\n', re.DOTALL)
    deprecated_extras_content = deprecated_extras_section_regex.findall(docs_content)[0]
    deprecated_extras_regexp = re.compile('\\|\\s(\\S+)\\s+\\|\\s(\\S*)\\s+\\|$', re.MULTILINE)
    for extras in deprecated_extras_regexp.findall(deprecated_extras_content):
        deprecated_extras[extras[0]] = extras[1]
    return deprecated_extras

def check_extras(console: Console) -> bool:
    if False:
        return 10
    '\n    Checks if non-deprecated extras match setup vs. doc.\n    :param console: print table there in case of errors\n    :return: True if all ok, False otherwise\n    '
    extras_table = Table()
    extras_table.add_column('NAME', justify='right', style='cyan')
    extras_table.add_column('SETUP', justify='center', style='magenta')
    extras_table.add_column('DOCS', justify='center', style='yellow')
    non_deprecated_setup_extras = get_extras_from_setup()
    non_deprecated_docs_extras = get_extras_from_docs()
    for extra in non_deprecated_setup_extras:
        if extra not in non_deprecated_docs_extras:
            extras_table.add_row(extra, 'V', '')
    for extra in non_deprecated_docs_extras:
        if extra not in non_deprecated_setup_extras:
            extras_table.add_row(extra, '', 'V')
    if extras_table.row_count != 0:
        print(f'[red bold]ERROR!![/red bold]\n\nThe "[bold]CORE_EXTRAS_DEPENDENCIES[/bold]"\nsections in the setup file: [bold yellow]{SETUP_PY_FILE}[/bold yellow]\nshould be synchronized with the "Extra Packages Reference"\nin the documentation file: [bold yellow]{DOCS_FILE}[/bold yellow].\n\nBelow is the list of extras that:\n\n  * are used but are not documented,\n  * are documented but not used,\n\n[bold]Please synchronize setup/documentation files![/bold]\n\n')
        console.print(extras_table)
        return False
    return True

def check_deprecated_extras(console: Console) -> bool:
    if False:
        while True:
            i = 10
    '\n    Checks if deprecated extras match setup vs. doc.\n    :param console: print table there in case of errors\n    :return: True if all ok, False otherwise\n    '
    deprecated_setup_extras = EXTRAS_DEPRECATED_ALIASES
    deprecated_docs_extras = get_deprecated_extras_from_docs()
    deprecated_extras_table = Table()
    deprecated_extras_table.add_column('DEPRECATED_IN_SETUP', justify='right', style='cyan')
    deprecated_extras_table.add_column('TARGET_IN_SETUP', justify='center', style='magenta')
    deprecated_extras_table.add_column('DEPRECATED_IN_DOCS', justify='right', style='cyan')
    deprecated_extras_table.add_column('TARGET_IN_DOCS', justify='center', style='magenta')
    for extra in deprecated_setup_extras.keys():
        if extra not in deprecated_docs_extras:
            deprecated_extras_table.add_row(extra, deprecated_setup_extras[extra], '', '')
        elif deprecated_docs_extras[extra] != deprecated_setup_extras[extra]:
            deprecated_extras_table.add_row(extra, deprecated_setup_extras[extra], extra, deprecated_docs_extras[extra])
    for extra in deprecated_docs_extras.keys():
        if extra not in deprecated_setup_extras:
            deprecated_extras_table.add_row('', '', extra, deprecated_docs_extras[extra])
    if deprecated_extras_table.row_count != 0:
        print(f'[red bold]ERROR!![/red bold]\n\nThe "[bold]EXTRAS_DEPRECATED_ALIASES[/bold]" section in the setup file:[bold yellow]{SETUP_PY_FILE}[/bold yellow]\nshould be synchronized with the "Extra Packages Reference"\nin the documentation file: [bold yellow]{DOCS_FILE}[/bold yellow].\n\nBelow is the list of deprecated extras that:\n\n  * are used but are not documented,\n  * are documented but not used,\n  * or have different target extra specified in the documentation or setup.\n\n[bold]Please synchronize setup/documentation files![/bold]\n\n')
        console.print(deprecated_extras_table)
        return False
    return True

def check_preinstalled_extras(console: Console) -> bool:
    if False:
        print('Hello World!')
    '\n    Checks if preinstalled extras match setup vs. doc.\n    :param console: print table there in case of errors\n    :return: True if all ok, False otherwise\n    '
    preinstalled_providers_from_docs = get_preinstalled_providers_from_docs()
    preinstalled_providers_from_setup = [provider.split('>=')[0] for provider in PREINSTALLED_PROVIDERS]
    preinstalled_providers_table = Table()
    preinstalled_providers_table.add_column('PREINSTALLED_IN_SETUP', justify='right', style='cyan')
    preinstalled_providers_table.add_column('PREINSTALLED_IN_DOCS', justify='center', style='magenta')
    for provider in preinstalled_providers_from_setup:
        if provider not in preinstalled_providers_from_docs:
            preinstalled_providers_table.add_row(provider, '')
    for provider in preinstalled_providers_from_docs:
        if provider not in preinstalled_providers_from_setup:
            preinstalled_providers_table.add_row('', provider)
    if preinstalled_providers_table.row_count != 0:
        print(f'[red bold]ERROR!![/red bold]\n\nThe "[bold]PREINSTALLED_PROVIDERS[/bold]" section in the setup file:[bold yellow]{SETUP_PY_FILE}[/bold yellow]\nshould be synchronized with the "Extra Packages Reference"\nin the documentation file: [bold yellow]{DOCS_FILE}[/bold yellow].\n\nBelow is the list of preinstalled providers that:\n  * are used but are not documented,\n  * or are documented but not used.\n\n[bold]Please synchronize setup/documentation files![/bold]\n\n')
        console.print(preinstalled_providers_table)
        return False
    return True
if __name__ == '__main__':
    status: list[bool] = []
    add_all_provider_packages()
    main_console = Console()
    status.append(check_extras(main_console))
    status.append(check_deprecated_extras(main_console))
    status.append(check_preinstalled_extras(main_console))
    if all(status):
        print('All extras are synchronized: [green]OK[/]')
        sys.exit(0)
    sys.exit(1)