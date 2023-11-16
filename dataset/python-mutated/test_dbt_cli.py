import shutil
from pathlib import Path
from dbt.include.starter_project import PACKAGE_PATH as starter_project_directory
from mage_ai.data_preparation.models.block.dbt.dbt_cli import DBTCli
from mage_ai.tests.base_test import TestCase

class DBTCliTest(TestCase):
    """
    Tests the DBTCli class, which is an interface with the dbt cli
    """

    @classmethod
    def setUpClass(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUpClass()
        self.project_name = 'dbt_test_project'
        self.project_dir = str(Path(self.repo_path) / self.project_name)
        shutil.copytree(starter_project_directory, self.project_dir, ignore=shutil.ignore_patterns(*['__init__.py', '__pycache__', '.gitkeep']))
        with (Path(self.project_dir) / 'dbt_project.yml').open('r+') as f:
            content = f'{f.read()}'.format(project_name=self.project_name, profile_name=self.project_name)
            f.seek(0)
            f.write(content)
            f.truncate()
        self.profiles_full_path = str(Path(self.project_dir) / 'profiles.yml')
        profiles_yaml = f"dbt_test_project:\n  outputs:\n   dev:\n     type: duckdb\n     path: {str(Path(self.project_dir) / 'test.db')}\n  target: dev\n"
        with Path(self.profiles_full_path).open('w') as f:
            f.write(profiles_yaml)
        self.model_full_path = str(Path(self.project_dir) / 'models' / 'mage_test_model.sql')
        model = "{{ config(materialized='table') }}select 1 as id"
        with Path(self.model_full_path).open('w') as f:
            f.write(model)
        self.schema_full_path = str(Path(self.project_dir) / 'models' / 'schema.yml')
        schema = '\nversion: 2\nmodels:\n  - name: mage_test_model\n    columns:\n      - name: id\n        tests:\n          - unique\n'
        with Path(self.schema_full_path).open('w') as f:
            f.write(schema)

    @classmethod
    def tearDownClass(self):
        if False:
            i = 10
            return i + 15
        shutil.rmtree(self.project_dir)
        super().tearDownClass()

    def test_invoke(self):
        if False:
            i = 10
            return i + 15
        DBTCli(['build', '--profiles-dir', self.project_dir, '--project-dir', self.project_dir, '--select', 'mage_test_model']).invoke()
        DBTCli(['clean', '--project-dir', self.project_dir]).invoke()

    def test_to_pandas(self):
        if False:
            return 10
        DBTCli(['run', '--profiles-dir', self.project_dir, '--project-dir', self.project_dir, '--select', 'mage_test_model']).invoke()
        (df, _res, success) = DBTCli(['show', '--profiles-dir', self.project_dir, '--project-dir', self.project_dir, '--select', 'mage_test_model', '--limit', '1']).to_pandas()
        self.assertTrue(success)
        self.assertEqual(df.to_dict(), {'id': {0: 1}})