import tempfile
from pathlib import Path
from textwrap import dedent
from tests.utils.cli_repo_creator import CliRunner, get_example_repo

def test_cli_apply_duplicated_featureview_names() -> None:
    if False:
        while True:
            i = 10
    run_simple_apply_test(example_repo_file_name='example_feature_repo_with_duplicated_featureview_names.py', expected_error=b'Please ensure that all feature view names are case-insensitively unique')

def test_cli_apply_duplicate_data_source_names() -> None:
    if False:
        while True:
            i = 10
    run_simple_apply_test(example_repo_file_name='example_repo_duplicate_data_source_names.py', expected_error=b'Multiple data sources share the same case-insensitive name')

def run_simple_apply_test(example_repo_file_name: str, expected_error: bytes):
    if False:
        i = 10
        return i + 15
    with tempfile.TemporaryDirectory() as repo_dir_name, tempfile.TemporaryDirectory() as data_dir_name:
        runner = CliRunner()
        repo_path = Path(repo_dir_name)
        data_path = Path(data_dir_name)
        repo_config = repo_path / 'feature_store.yaml'
        repo_config.write_text(dedent(f"\n        project: foo\n        registry: {data_path / 'registry.db'}\n        provider: local\n        online_store:\n            path: {data_path / 'online_store.db'}\n        "))
        repo_example = repo_path / 'example.py'
        repo_example.write_text(get_example_repo(example_repo_file_name))
        (rc, output) = runner.run_with_output(['apply'], cwd=repo_path)
        assert rc != 0 and expected_error in output

def test_cli_apply_imported_featureview() -> None:
    if False:
        while True:
            i = 10
    '\n    Tests that applying a feature view imported from a separate Python file is successful.\n    '
    with tempfile.TemporaryDirectory() as repo_dir_name, tempfile.TemporaryDirectory() as data_dir_name:
        runner = CliRunner()
        repo_path = Path(repo_dir_name)
        data_path = Path(data_dir_name)
        repo_config = repo_path / 'feature_store.yaml'
        repo_config.write_text(dedent(f"\n        project: foo\n        registry: {data_path / 'registry.db'}\n        provider: local\n        online_store:\n            path: {data_path / 'online_store.db'}\n        "))
        repo_example = repo_path / 'example.py'
        repo_example.write_text(get_example_repo('example_feature_repo_with_driver_stats_feature_view.py'))
        repo_example_2 = repo_path / 'example_2.py'
        repo_example_2.write_text("from example import driver_hourly_stats_view\nfrom feast import FeatureService\na_feature_service = FeatureService(\n   name='driver_locations_service',\n   features=[driver_hourly_stats_view],\n)\n")
        (rc, output) = runner.run_with_output(['apply'], cwd=repo_path)
        assert rc == 0
        assert b'Created feature service driver_locations_service' in output

def test_cli_apply_imported_featureview_with_duplication() -> None:
    if False:
        while True:
            i = 10
    '\n    Tests that applying feature views with duplicated names is not possible, even if one of the\n    duplicated feature views is imported from another file.\n    '
    with tempfile.TemporaryDirectory() as repo_dir_name, tempfile.TemporaryDirectory() as data_dir_name:
        runner = CliRunner()
        repo_path = Path(repo_dir_name)
        data_path = Path(data_dir_name)
        repo_config = repo_path / 'feature_store.yaml'
        repo_config.write_text(dedent(f"\n        project: foo\n        registry: {data_path / 'registry.db'}\n        provider: local\n        online_store:\n            path: {data_path / 'online_store.db'}\n        "))
        repo_example = repo_path / 'example.py'
        repo_example.write_text(get_example_repo('example_feature_repo_with_driver_stats_feature_view.py'))
        repo_example_2 = repo_path / 'example_2.py'
        repo_example_2.write_text("from datetime import timedelta\nfrom example import driver, driver_hourly_stats, driver_hourly_stats_view\nfrom feast import FeatureService, FeatureView\na_feature_service = FeatureService(\n   name='driver_locations_service',\n   features=[driver_hourly_stats_view],\n)\ndriver_hourly_stats_view_2 = FeatureView(\n   name='driver_hourly_stats',\n   entities=[driver],\n   ttl=timedelta(days=1),\n   online=True,\n   source=driver_hourly_stats,\n   tags={'dummy': 'true'})\n")
        (rc, output) = runner.run_with_output(['apply'], cwd=repo_path)
        assert rc != 0
        assert b'More than one feature view with name driver_hourly_stats found.' in output

def test_cli_apply_duplicated_featureview_names_multiple_py_files() -> None:
    if False:
        while True:
            i = 10
    '\n    Test apply feature views with duplicated names from multiple py files in a feature repo using CLI\n    '
    with tempfile.TemporaryDirectory() as repo_dir_name, tempfile.TemporaryDirectory() as data_dir_name:
        runner = CliRunner()
        repo_path = Path(repo_dir_name)
        data_path = Path(data_dir_name)
        repo_config = repo_path / 'feature_store.yaml'
        repo_config.write_text(dedent(f"\n        project: foo\n        registry: {data_path / 'registry.db'}\n        provider: local\n        online_store:\n            path: {data_path / 'online_store.db'}\n        "))
        for i in range(3):
            repo_example = repo_path / f'example{i}.py'
            repo_example.write_text(get_example_repo('example_feature_repo_with_driver_stats_feature_view.py'))
        (rc, output) = runner.run_with_output(['apply'], cwd=repo_path)
        assert rc != 0 and b'Please ensure that all feature view names are case-insensitively unique' in output