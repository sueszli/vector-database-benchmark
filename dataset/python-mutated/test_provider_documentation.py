from __future__ import annotations
from typing import Iterable
import pytest
from airflow_breeze.prepare_providers.provider_documentation import Change, _convert_git_changes_to_table, _convert_pip_requirements_to_table, _find_insertion_index_for_version, _get_change_from_line, _get_changes_classified, _get_git_log_command, _get_version_tag, _verify_changelog_exists
from airflow_breeze.utils.packages import get_pip_package_name, get_wheel_package_name
from airflow_breeze.utils.path_utils import AIRFLOW_SOURCES_ROOT
CHANGELOG_CONTENT = "\nChangelog\n---------\n\n5.0.0\n.....\n\nBreaking changes\n~~~~~~~~~~~~~~~~\n\nThe ``offset`` parameter has been deprecated from ``list_jobs`` in favor of faster pagination with ``page_token`` similarly to `Databricks API <https://docs.databricks.com/api/workspace/jobs/list>`_.\n\n* ``Remove offset-based pagination from 'list_jobs' function in 'DatabricksHook' (#34926)``\n\n4.7.0\n.....\n\nFeatures\n~~~~~~~~\n\n* ``Add operator to create jobs in Databricks (#35156)``\n\n.. Below changes are excluded from the changelog. Move them to\n   appropriate section above if needed. Do not delete the lines(!):\n   * ``Prepare docs 3rd wave of Providers October 2023 (#35187)``\n   * ``Pre-upgrade 'ruff==0.0.292' changes in providers (#35053)``\n   * ``D401 Support - Providers: DaskExecutor to Github (Inclusive) (#34935)``\n\n4.6.0\n.....\n\n.. note::\n  This release of provider is only available for Airflow 2.5+ as explained in the\n  `Apache Airflow providers support policy <https://github.com/apache/airflow/blob/main/PROVIDERS.rst#minimum-supported-version-of-airflow-for-community-managed-providers>`_.\n\n"

def test_find_insertion_index_append_to_found_changelog():
    if False:
        i = 10
        return i + 15
    (index, append) = _find_insertion_index_for_version(CHANGELOG_CONTENT.splitlines(), '5.0.0')
    assert append
    assert index == 13

def test_find_insertion_index_insert_new_changelog():
    if False:
        while True:
            i = 10
    (index, append) = _find_insertion_index_for_version(CHANGELOG_CONTENT.splitlines(), '5.0.1')
    assert not append
    assert index == 3

@pytest.mark.parametrize('version, provider_id, suffix, tag', [('1.0.1', 'asana', '', 'providers-asana/1.0.1'), ('1.0.1', 'asana', 'rc1', 'providers-asana/1.0.1rc1'), ('1.0.1', 'apache.hdfs', 'beta1', 'providers-apache-hdfs/1.0.1beta1')])
def test_get_version_tag(version: str, provider_id: str, suffix: str, tag: str):
    if False:
        print('Hello World!')
    assert _get_version_tag(version, provider_id, suffix) == tag

@pytest.mark.parametrize('from_commit, to_commit, git_command', [(None, None, ['git', 'log', '--pretty=format:%H %h %cd %s', '--date=short', '--', '.']), ('from_tag', None, ['git', 'log', '--pretty=format:%H %h %cd %s', '--date=short', 'from_tag', '--', '.']), ('from_tag', 'to_tag', ['git', 'log', '--pretty=format:%H %h %cd %s', '--date=short', 'from_tag...to_tag', '--', '.'])])
def test_get_git_log_command(from_commit: str | None, to_commit: str | None, git_command: list[str]):
    if False:
        print('Hello World!')
    assert _get_git_log_command(from_commit, to_commit) == git_command

def test_get_git_log_command_wrong():
    if False:
        return 10
    with pytest.raises(ValueError, match='to_commit without from_commit'):
        _get_git_log_command(None, 'to_commit')

@pytest.mark.parametrize('provider_id, pip_package_name', [('asana', 'apache-airflow-providers-asana'), ('apache.hdfs', 'apache-airflow-providers-apache-hdfs')])
def test_get_pip_package_name(provider_id: str, pip_package_name: str):
    if False:
        i = 10
        return i + 15
    assert get_pip_package_name(provider_id) == pip_package_name

@pytest.mark.parametrize('provider_id, wheel_package_name', [('asana', 'apache_airflow_providers_asana'), ('apache.hdfs', 'apache_airflow_providers_apache_hdfs')])
def test_get_wheel_package_name(provider_id: str, wheel_package_name: str):
    if False:
        i = 10
        return i + 15
    assert get_wheel_package_name(provider_id) == wheel_package_name

@pytest.mark.parametrize('line, version, change', [('LONG_HASH_123144 SHORT_HASH 2023-01-01 Description `with` no pr', '1.0.1', Change(full_hash='LONG_HASH_123144', short_hash='SHORT_HASH', date='2023-01-01', version='1.0.1', message='Description `with` no pr', message_without_backticks="Description 'with' no pr", pr=None)), ('LONG_HASH_123144 SHORT_HASH 2023-01-01 Description `with` pr (#12345)', '1.0.1', Change(full_hash='LONG_HASH_123144', short_hash='SHORT_HASH', date='2023-01-01', version='1.0.1', message='Description `with` pr (#12345)', message_without_backticks="Description 'with' pr (#12345)", pr='12345'))])
def test_get_change_from_line(line: str, version: str, change: Change):
    if False:
        for i in range(10):
            print('nop')
    assert _get_change_from_line(line, version) == change

@pytest.mark.parametrize('input, output, markdown, changes_len', [('\nLONG_HASH_123144 SHORT_HASH 2023-01-01 Description `with` no pr\nLONG_HASH_123144 SHORT_HASH 2023-01-01 Description `with` pr (#12345)\n\nLONG_HASH_123144 SHORT_HASH 2023-01-01 Description `with` pr (#12346)\n\n', "\n1.0.1\n.....\n\nLatest change: 2023-01-01\n\n============================================  ===========  ==================================\nCommit                                        Committed    Subject\n============================================  ===========  ==================================\n`SHORT_HASH <https://url/LONG_HASH_123144>`_  2023-01-01   ``Description 'with' no pr``\n`SHORT_HASH <https://url/LONG_HASH_123144>`_  2023-01-01   ``Description 'with' pr (#12345)``\n`SHORT_HASH <https://url/LONG_HASH_123144>`_  2023-01-01   ``Description 'with' pr (#12346)``\n============================================  ===========  ==================================", False, 3), ('\nLONG_HASH_123144 SHORT_HASH 2023-01-01 Description `with` no pr\nLONG_HASH_123144 SHORT_HASH 2023-01-01 Description `with` pr (#12345)\n\nLONG_HASH_123144 SHORT_HASH 2023-01-01 Description `with` pr (#12346)\n\n', "\n| Commit                                     | Committed   | Subject                          |\n|:-------------------------------------------|:------------|:---------------------------------|\n| [SHORT_HASH](https://url/LONG_HASH_123144) | 2023-01-01  | `Description 'with' no pr`       |\n| [SHORT_HASH](https://url/LONG_HASH_123144) | 2023-01-01  | `Description 'with' pr (#12345)` |\n| [SHORT_HASH](https://url/LONG_HASH_123144) | 2023-01-01  | `Description 'with' pr (#12346)` |\n", True, 3)])
def test_convert_git_changes_to_table(input: str, output: str, markdown: bool, changes_len):
    if False:
        print('Hello World!')
    (table, list_of_changes) = _convert_git_changes_to_table(version='1.0.1', changes=input, base_url='https://url/', markdown=markdown)
    assert table.strip() == output.strip()
    assert len(list_of_changes) == changes_len
    assert list_of_changes[0].pr is None
    assert list_of_changes[1].pr == '12345'
    assert list_of_changes[2].pr == '12346'

@pytest.mark.parametrize('requirements, markdown, table', [(['apache-airflow>2.5.0'], False, '\n==================  ==================\nPIP package         Version required\n==================  ==================\n``apache-airflow``  ``>2.5.0``\n==================  ==================\n'), (['apache-airflow>2.5.0'], True, '\n| PIP package      | Version required   |\n|:-----------------|:-------------------|\n| `apache-airflow` | `>2.5.0`           |\n')])
def test_convert_pip_requirements_to_table(requirements: Iterable[str], markdown: bool, table: str):
    if False:
        print('Hello World!')
    print(_convert_pip_requirements_to_table(requirements, markdown))
    assert _convert_pip_requirements_to_table(requirements, markdown).strip() == table.strip()

def test_verify_changelog_exists():
    if False:
        for i in range(10):
            print('nop')
    assert _verify_changelog_exists('asana') == AIRFLOW_SOURCES_ROOT / 'airflow' / 'providers' / 'asana' / 'CHANGELOG.rst'

@pytest.mark.parametrize('descriptions, with_breaking_changes, maybe_with_new_features,breaking_count, feature_count, bugfix_count, other_count', [(['Added feature x'], True, True, 0, 1, 0, 0), (['Added feature x'], False, False, 0, 0, 0, 1), (['Breaking change in'], True, True, 1, 0, 0, 0), (['Breaking change in', 'Added feature y'], True, True, 1, 1, 0, 0), (['Fix change in', 'Breaking feature y'], False, True, 0, 0, 1, 1), (['Fix change in', 'Breaking feature y'], False, True, 0, 0, 1, 1)])
def test_classify_changes_automatically(descriptions: list[str], with_breaking_changes: bool, maybe_with_new_features: bool, breaking_count: int, feature_count: int, bugfix_count: int, other_count: int):
    if False:
        print('Hello World!')
    'Test simple automated classification of the changes based on their single-line description.'
    changes = [_get_change_from_line(f'LONG SHORT 2023-12-01 {description}', version='0.1.0') for description in descriptions]
    classified_changes = _get_changes_classified(changes, with_breaking_changes=with_breaking_changes, maybe_with_new_features=maybe_with_new_features)
    assert len(classified_changes.breaking_changes) == breaking_count
    assert len(classified_changes.features) == feature_count
    assert len(classified_changes.fixes) == bugfix_count
    assert len(classified_changes.other) == other_count