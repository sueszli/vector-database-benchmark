import sys
from pathlib import Path
from typing import Callable, Iterable, Optional, Set, Tuple
from connector_ops.utils import Connector, ConnectorLanguage
from pydash.objects import get

def check_migration_guide(connector: Connector) -> bool:
    if False:
        i = 10
        return i + 15
    'Check if a migration guide is available for the connector if a breaking change was introduced.'
    breaking_changes = get(connector.metadata, 'releases.breakingChanges')
    if not breaking_changes:
        return True
    migration_guide_file_path = connector.migration_guide_file_path
    migration_guide_exists = migration_guide_file_path is not None and migration_guide_file_path.exists()
    if not migration_guide_exists:
        print(f'Migration guide file is missing for {connector.name}. Please create a migration guide at {connector.migration_guide_file_path}')
        return False
    expected_title = f'# {connector.name_from_metadata} Migration Guide'
    expected_version_header_start = '## Upgrading to '
    with open(migration_guide_file_path) as f:
        first_line = f.readline().strip()
        if not first_line == expected_title:
            print(f"Migration guide file for {connector.technical_name} does not start with the correct header. Expected '{expected_title}', got '{first_line}'")
            return False
        ordered_breaking_changes = sorted(breaking_changes.keys(), reverse=True)
        ordered_expected_headings = [f'{expected_version_header_start}{version}' for version in ordered_breaking_changes]
        ordered_heading_versions = []
        for line in f:
            stripped_line = line.strip()
            if stripped_line.startswith(expected_version_header_start):
                version = stripped_line.replace(expected_version_header_start, '')
                ordered_heading_versions.append(version)
        if ordered_breaking_changes != ordered_heading_versions:
            print(f'Migration guide file for {connector.name} has incorrect version headings.')
            print('Check for missing, extra, or misordered headings, or headers with typos.')
            print(f'Expected headings: {ordered_expected_headings}')
            return False
    return True

def check_documentation_file_exists(connector: Connector) -> bool:
    if False:
        while True:
            i = 10
    'Check if a markdown file with connector documentation is available\n    in docs/integrations/<connector-type>s/<connector-name>.md\n\n    Args:\n        connector (Connector): a Connector dataclass instance.\n\n    Returns:\n        bool: Wether a documentation file was found.\n    '
    file_path = connector.documentation_file_path
    return file_path is not None and file_path.exists()

def check_documentation_follows_guidelines(connector: Connector) -> bool:
    if False:
        while True:
            i = 10
    'Documentation guidelines are defined here https://hackmd.io/Bz75cgATSbm7DjrAqgl4rw'
    follows_guidelines = True
    with open(connector.documentation_file_path) as f:
        doc_lines = [line.lower() for line in f.read().splitlines()]
    if not doc_lines[0].startswith('# '):
        print('The connector name is not used as the main header in the documentation.')
        follows_guidelines = False
    if connector.metadata:
        if doc_lines[0].strip() != f"# {connector.metadata['name'].lower()}":
            print('The connector name is not used as the main header in the documentation.')
            follows_guidelines = False
    elif not doc_lines[0].startswith('# '):
        print('The connector name is not used as the main header in the documentation.')
        follows_guidelines = False
    expected_sections = ['## Prerequisites', '## Setup guide', '## Supported sync modes', '## Supported streams', '## Changelog']
    for expected_section in expected_sections:
        if expected_section.lower() not in doc_lines:
            print(f"Connector documentation is missing a '{expected_section.replace('#', '').strip()}' section.")
            follows_guidelines = False
    return follows_guidelines

def check_changelog_entry_is_updated(connector: Connector) -> bool:
    if False:
        while True:
            i = 10
    'Check that the changelog entry is updated for the latest connector version\n    in docs/integrations/<connector-type>/<connector-name>.md\n\n    Args:\n        connector (Connector): a Connector dataclass instance.\n\n    Returns:\n        bool: Wether a the changelog is up to date.\n    '
    if not check_documentation_file_exists(connector):
        return False
    with open(connector.documentation_file_path) as f:
        after_changelog = False
        for line in f:
            if '# changelog' in line.lower():
                after_changelog = True
            if after_changelog and connector.version in line:
                return True
    return False

def check_connector_icon_is_available(connector: Connector) -> bool:
    if False:
        return 10
    'Check an SVG icon exists for a connector in\n    in airbyte-config-oss/init-oss/src/main/resources/icons/<connector-name>.svg\n\n    Args:\n        connector (Connector): a Connector dataclass instance.\n\n    Returns:\n        bool: Wether an icon exists for this connector.\n    '
    return connector.icon_path.exists()

def read_all_files_in_directory(directory: Path, ignored_directories: Optional[Set[str]]=None, ignored_filename_patterns: Optional[Set[str]]=None) -> Iterable[Tuple[str, str]]:
    if False:
        i = 10
        return i + 15
    ignored_directories = ignored_directories if ignored_directories is not None else {}
    ignored_filename_patterns = ignored_filename_patterns if ignored_filename_patterns is not None else {}
    for path in directory.rglob('*'):
        ignore_directory = any([ignored_directory in path.parts for ignored_directory in ignored_directories])
        ignore_filename = any([path.match(ignored_filename_pattern) for ignored_filename_pattern in ignored_filename_patterns])
        ignore = ignore_directory or ignore_filename
        if path.is_file() and (not ignore):
            try:
                for line in open(path, 'r'):
                    yield (path, line)
            except UnicodeDecodeError:
                print(f'{path} could not be decoded as it is not UTF8.')
                continue
IGNORED_DIRECTORIES_FOR_HTTPS_CHECKS = {'.venv', 'tests', 'unit_tests', 'integration_tests', 'build', 'source-file', '.pytest_cache', 'acceptance_tests_logs', '.hypothesis'}
IGNORED_FILENAME_PATTERN_FOR_HTTPS_CHECKS = {'*Test.java', '*.jar', '*.pyc', '*.gz', '*.svg', 'expected_records.jsonl', 'expected_records.json'}
IGNORED_URLS_PREFIX = {'http://json-schema.org', 'http://localhost'}

def is_comment(line: str, file_path: Path):
    if False:
        print('Hello World!')
    language_comments = {'.py': '#', '.yml': '#', '.yaml': '#', '.java': '//', '.md': '<!--'}
    denote_comment = language_comments.get(file_path.suffix)
    if not denote_comment:
        return False
    trimmed_line = line.lstrip()
    return trimmed_line.startswith(denote_comment)

def check_connector_https_url_only(connector: Connector) -> bool:
    if False:
        i = 10
        return i + 15
    'Check a connector code contains only https url.\n\n    Args:\n        connector (Connector): a Connector dataclass instance.\n\n    Returns:\n        bool: Wether the connector code contains only https url.\n    '
    files_with_http_url = set()
    ignore_comment = '# ignore-https-check'
    for (filename, line) in read_all_files_in_directory(connector.code_directory, IGNORED_DIRECTORIES_FOR_HTTPS_CHECKS, IGNORED_FILENAME_PATTERN_FOR_HTTPS_CHECKS):
        line = line.lower()
        if is_comment(line, filename):
            continue
        if ignore_comment in line:
            continue
        for prefix in IGNORED_URLS_PREFIX:
            line = line.replace(prefix, '')
        if 'http://' in line:
            files_with_http_url.add(str(filename))
    if files_with_http_url:
        files_with_http_url = '\n\t- '.join(files_with_http_url)
        print(f'The following files have http:// URLs:\n\t- {files_with_http_url}')
        return False
    return True

def check_connector_has_no_critical_vulnerabilities(connector: Connector) -> bool:
    if False:
        print('Hello World!')
    'Check if the connector image is free of critical Snyk vulnerabilities.\n    Runs a docker scan command.\n\n    Args:\n        connector (Connector): a Connector dataclass instance.\n\n    Returns:\n        bool: Wether the connector is free of critical vulnerabilities.\n    '
    return True

def check_metadata_version_matches_dockerfile_label(connector: Connector) -> bool:
    if False:
        return 10
    version_in_dockerfile = connector.version_in_dockerfile_label
    if version_in_dockerfile is None:
        return connector.language == ConnectorLanguage.JAVA
    return version_in_dockerfile == connector.version
DEFAULT_QA_CHECKS = (check_documentation_file_exists, check_migration_guide, check_changelog_entry_is_updated, check_connector_icon_is_available, check_connector_https_url_only, check_connector_has_no_critical_vulnerabilities)

def get_qa_checks_to_run(connector: Connector) -> Tuple[Callable]:
    if False:
        print('Hello World!')
    if connector.has_dockerfile:
        return DEFAULT_QA_CHECKS + (check_metadata_version_matches_dockerfile_label,)
    return DEFAULT_QA_CHECKS

def remove_strict_encrypt_suffix(connector_technical_name: str) -> str:
    if False:
        while True:
            i = 10
    'Remove the strict encrypt suffix from a connector name.\n\n    Args:\n        connector_technical_name (str): the connector name.\n\n    Returns:\n        str: the connector name without the strict encrypt suffix.\n    '
    strict_encrypt_suffixes = ['-strict-encrypt', '-secure']
    for suffix in strict_encrypt_suffixes:
        if connector_technical_name.endswith(suffix):
            new_connector_technical_name = connector_technical_name.replace(suffix, '')
            print('Checking connector ' + new_connector_technical_name + ' due to strict-encrypt')
            return new_connector_technical_name
    return connector_technical_name

def run_qa_checks():
    if False:
        i = 10
        return i + 15
    connector_technical_name = sys.argv[1].split('/')[-1]
    if not connector_technical_name.startswith('source-') and (not connector_technical_name.startswith('destination-')):
        print('No QA check to run as this is not a connector.')
        sys.exit(0)
    connector_technical_name = remove_strict_encrypt_suffix(connector_technical_name)
    connector = Connector(connector_technical_name)
    print(f'Running QA checks for {connector_technical_name}:{connector.version}')
    qa_check_results = {qa_check.__name__: qa_check(connector) for qa_check in get_qa_checks_to_run(connector)}
    if not all(qa_check_results.values()):
        print(f'QA checks failed for {connector_technical_name}:{connector.version}:')
        for (check_name, check_result) in qa_check_results.items():
            check_result_prefix = '✅' if check_result else '❌'
            print(f'{check_result_prefix} - {check_name}')
        sys.exit(1)
    else:
        print(f'All QA checks succeeded for {connector_technical_name}:{connector.version}')
        sys.exit(0)