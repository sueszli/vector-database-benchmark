from __future__ import annotations
import functools
import sys
from pathlib import Path
from pprint import pprint
import requests
import semver
import yaml
from packaging import version
ROOT_DIR = Path(__file__).resolve().parent / '..'
KNOWN_FALSE_DETECTIONS = {('logging', 'extra_logger_names', '2.2.0'), ('core', 'mp_start_method', '2.5.1')}
RENAMED_SECTIONS = [('kubernetes_executor', 'kubernetes', '2.4.3')]
CONFIG_TEMPLATE_FORMAT_UPDATE = '2.6.0'

def fetch_pypi_versions() -> list[str]:
    if False:
        i = 10
        return i + 15
    r = requests.get('https://pypi.org/pypi/apache-airflow/json')
    r.raise_for_status()
    all_version = r.json()['releases'].keys()
    released_versions = [d for d in all_version if not ('rc' in d or 'b' in d)]
    return released_versions

def parse_config_template_new_format(config_content: str) -> set[tuple[str, str, str]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Parses config_template.yaml new format and returns config_options\n    '
    config_sections = yaml.safe_load(config_content)
    return {(config_section_name, config_option_name, config_option_value['version_added']) for (config_section_name, config_section_value) in config_sections.items() for (config_option_name, config_option_value) in config_section_value['options'].items()}

def parse_config_template_old_format(config_content: str) -> set[tuple[str, str, str]]:
    if False:
        return 10
    '\n    Parses config_template.yaml old format and returns config_options\n    '
    config_sections = yaml.safe_load(config_content)
    return {(config_section['name'], config_option['name'], config_option.get('version_added')) for config_section in config_sections for config_option in config_section['options']}

@functools.lru_cache
def fetch_config_options_for_version(version_str: str) -> set[tuple[str, str]]:
    if False:
        print('Hello World!')
    r = requests.get(f'https://raw.githubusercontent.com/apache/airflow/{version_str}/airflow/config_templates/config.yml')
    r.raise_for_status()
    content = r.text
    if version.parse(version_str) >= version.parse(CONFIG_TEMPLATE_FORMAT_UPDATE):
        config_options = parse_config_template_new_format(content)
    else:
        config_options = parse_config_template_old_format(content)
    return {(section_name, option_name) for (section_name, option_name, _) in config_options}

def read_local_config_options() -> set[tuple[str, str, str]]:
    if False:
        i = 10
        return i + 15
    return parse_config_template_new_format((ROOT_DIR / 'airflow' / 'config_templates' / 'config.yml').read_text())
computed_option_new_section = set()
for (new_section, old_section, version_before_renaming) in RENAMED_SECTIONS:
    options = fetch_config_options_for_version(version_before_renaming)
    options = {(new_section, option_name) for (section_name, option_name) in options if section_name == old_section}
    computed_option_new_section.update(options)
to_check_versions: list[str] = [d for d in fetch_pypi_versions() if d.startswith('2.')]
to_check_versions.sort(key=semver.VersionInfo.parse)
expected_computed_options: set[tuple[str, str, str]] = set()
for (prev_version, curr_version) in zip(to_check_versions[:-1], to_check_versions[1:]):
    print('Processing version:', curr_version)
    options_1 = fetch_config_options_for_version(prev_version)
    options_2 = fetch_config_options_for_version(curr_version)
    new_options = options_2 - options_1
    new_options -= computed_option_new_section
    expected_computed_options.update({(section_name, option_name, curr_version) for (section_name, option_name) in new_options})
print('Expected computed options count:', len(expected_computed_options))
local_options = read_local_config_options()
print('Local options count:', len(local_options))
local_options_plain: set[tuple[str, str]] = {(section_name, option_name) for (section_name, option_name, version_added) in local_options}
computed_options: set[tuple[str, str, str]] = {(section_name, option_name, version_added) for (section_name, option_name, version_added) in expected_computed_options if (section_name, option_name) in local_options_plain}
print('Visible computed options count:', len(computed_options))
local_options_with_version_added: set[tuple[str, str, str]] = {(section_name, option_name, version_added) for (section_name, option_name, version_added) in local_options if version_added}
diff_options: set[tuple[str, str, str]] = computed_options - local_options_with_version_added
diff_options -= KNOWN_FALSE_DETECTIONS
if diff_options:
    pprint(diff_options)
    sys.exit(1)
else:
    print('No changes required')