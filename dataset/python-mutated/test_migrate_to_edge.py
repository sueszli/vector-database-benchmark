import pytest
from django.core.management import call_command
from django.core.management.base import CommandError
from environments.dynamodb.migrator import IdentityMigrator

def test_calling_migrate_to_edge_calls_migrate_identities_with_correct_arguments(mocker):
    if False:
        i = 10
        return i + 15
    project_id = 1
    mocked_identity_migrator = mocker.patch('environments.management.commands.migrate_to_edge.IdentityMigrator', spec=IdentityMigrator)
    mocked_identity_migrator.return_value.can_migrate = True
    call_command('migrate_to_edge', project_id)
    mocked_identity_migrator.assert_called_with(project_id)
    mocked_identity_migrator.return_value.migrate.assert_called_with()

def test_calling_migrate_to_edge_raises_command_error_if_identities_are_already_migrated(mocker):
    if False:
        while True:
            i = 10
    project_id = 1
    mocked_identity_migrator = mocker.patch('environments.management.commands.migrate_to_edge.IdentityMigrator', spec=IdentityMigrator)
    mocked_identity_migrator.return_value.can_migrate = False
    with pytest.raises(CommandError):
        call_command('migrate_to_edge', project_id)
    mocked_identity_migrator.assert_called_with(project_id)
    mocked_identity_migrator.return_value.migrate.assert_not_called()