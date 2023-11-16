"""Remove ``can_read`` permission on config resource for ``User`` and ``Viewer`` role

Revision ID: 82b7c48c147f
Revises: e959f08ac86c
Create Date: 2021-02-04 12:45:58.138224

"""
from __future__ import annotations
import logging
from airflow.security import permissions
from airflow.www.app import cached_app
revision = '82b7c48c147f'
down_revision = 'e959f08ac86c'
branch_labels = None
depends_on = None
airflow_version = '2.0.1'

def upgrade():
    if False:
        return 10
    'Remove can_read action from config resource for User and Viewer role'
    log = logging.getLogger()
    handlers = log.handlers[:]
    appbuilder = cached_app(config={'FAB_UPDATE_PERMS': False}).appbuilder
    roles_to_modify = [role for role in appbuilder.sm.get_all_roles() if role.name in ['User', 'Viewer']]
    can_read_on_config_perm = appbuilder.sm.get_permission(permissions.ACTION_CAN_READ, permissions.RESOURCE_CONFIG)
    for role in roles_to_modify:
        if appbuilder.sm.permission_exists_in_one_or_more_roles(permissions.RESOURCE_CONFIG, permissions.ACTION_CAN_READ, [role.id]):
            appbuilder.sm.remove_permission_from_role(role, can_read_on_config_perm)
    log.handlers = handlers

def downgrade():
    if False:
        print('Hello World!')
    'Add can_read action on config resource for User and Viewer role'
    appbuilder = cached_app(config={'FAB_UPDATE_PERMS': False}).appbuilder
    roles_to_modify = [role for role in appbuilder.sm.get_all_roles() if role.name in ['User', 'Viewer']]
    can_read_on_config_perm = appbuilder.sm.get_permission(permissions.ACTION_CAN_READ, permissions.RESOURCE_CONFIG)
    for role in roles_to_modify:
        if not appbuilder.sm.permission_exists_in_one_or_more_roles(permissions.RESOURCE_CONFIG, permissions.ACTION_CAN_READ, [role.id]):
            appbuilder.sm.add_permission_to_role(role, can_read_on_config_perm)