from __future__ import annotations
from unittest import mock
import pytest
from airflow.auth.managers.fab.cli_commands import sync_perm_command
from airflow.cli import cli_parser
pytestmark = pytest.mark.db_test

class TestCliSyncPerm:

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        cls.parser = cli_parser.get_parser()

    @mock.patch('airflow.auth.managers.fab.cli_commands.utils.get_application_builder')
    def test_cli_sync_perm(self, mock_get_application_builder):
        if False:
            for i in range(10):
                print('nop')
        mock_appbuilder = mock.MagicMock()
        mock_get_application_builder.return_value.__enter__.return_value = mock_appbuilder
        args = self.parser.parse_args(['sync-perm'])
        sync_perm_command.sync_perm(args)
        mock_appbuilder.add_permissions.assert_called_once_with(update_perms=True)
        mock_appbuilder.sm.sync_roles.assert_called_once_with()
        mock_appbuilder.sm.create_dag_specific_permissions.assert_not_called()

    @mock.patch('airflow.auth.managers.fab.cli_commands.utils.get_application_builder')
    def test_cli_sync_perm_include_dags(self, mock_get_application_builder):
        if False:
            for i in range(10):
                print('nop')
        mock_appbuilder = mock.MagicMock()
        mock_get_application_builder.return_value.__enter__.return_value = mock_appbuilder
        args = self.parser.parse_args(['sync-perm', '--include-dags'])
        sync_perm_command.sync_perm(args)
        mock_appbuilder.add_permissions.assert_called_once_with(update_perms=True)
        mock_appbuilder.sm.sync_roles.assert_called_once_with()
        mock_appbuilder.sm.create_dag_specific_permissions.assert_called_once_with()