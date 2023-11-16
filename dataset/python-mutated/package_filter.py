from __future__ import annotations
import fnmatch
from pathlib import Path
PROVIDERS_DIR = Path(__file__).parents[3].resolve() / 'airflow' / 'providers'

def get_removed_provider_ids() -> list[str]:
    if False:
        print('Hello World!')
    '\n    Yields the ids of suspended providers.\n    '
    import yaml
    removed_provider_ids = []
    for provider_path in PROVIDERS_DIR.rglob('provider.yaml'):
        provider_yaml = yaml.safe_load(provider_path.read_text())
        if provider_yaml.get('removed'):
            removed_provider_ids.append(provider_yaml['package-name'][len('apache-airflow-providers-'):].replace('-', '.'))
    return removed_provider_ids

def process_package_filters(available_packages: list[str], package_filters: list[str] | None):
    if False:
        i = 10
        return i + 15
    'Filters the package list against a set of filters.\n\n    A packet is returned if it matches at least one filter. The function keeps the order of the packages.\n    '
    if not package_filters:
        return available_packages
    suspended_packages = [f"apache-airflow-providers-{provider.replace('.', '-')}" for provider in get_removed_provider_ids()]
    all_packages_with_suspended = available_packages + suspended_packages
    invalid_filters = [f for f in package_filters if not any((fnmatch.fnmatch(p, f) for p in all_packages_with_suspended))]
    if invalid_filters:
        raise SystemExit(f'Some filters did not find any package: {invalid_filters}, Please check if they are correct.')
    return [p for p in all_packages_with_suspended if any((fnmatch.fnmatch(p, f) for f in package_filters))]