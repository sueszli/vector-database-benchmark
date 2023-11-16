"""Sync permission command."""
from __future__ import annotations
from airflow.utils import cli as cli_utils
from airflow.utils.providers_configuration_loader import providers_configuration_loaded

@cli_utils.action_cli
@providers_configuration_loaded
def sync_perm(args):
    if False:
        i = 10
        return i + 15
    'Update permissions for existing roles and DAGs.'
    from airflow.auth.managers.fab.cli_commands.utils import get_application_builder
    with get_application_builder() as appbuilder:
        print('Updating actions and resources for all existing roles')
        appbuilder.add_permissions(update_perms=True)
        appbuilder.sm.sync_roles()
        if args.include_dags:
            print('Updating permission on all DAG views')
            appbuilder.sm.create_dag_specific_permissions()