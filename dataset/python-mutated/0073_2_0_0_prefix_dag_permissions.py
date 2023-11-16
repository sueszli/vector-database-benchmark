"""Prefix DAG permissions.

Revision ID: 849da589634d
Revises: 45ba3f1493b9
Create Date: 2020-10-01 17:25:10.006322

"""
from __future__ import annotations
from flask_appbuilder import SQLA
from airflow import settings
from airflow.security import permissions
from airflow.auth.managers.fab.models import Action, Permission, Resource
revision = '849da589634d'
down_revision = '45ba3f1493b9'
branch_labels = None
depends_on = None
airflow_version = '2.0.0'

def prefix_individual_dag_permissions(session):
    if False:
        return 10
    dag_perms = ['can_dag_read', 'can_dag_edit']
    prefix = 'DAG:'
    perms = session.query(Permission).join(Action).filter(Action.name.in_(dag_perms)).join(Resource).filter(Resource.name != 'all_dags').filter(Resource.name.notlike(prefix + '%')).all()
    resource_ids = {permission.resource.id for permission in perms}
    vm_query = session.query(Resource).filter(Resource.id.in_(resource_ids))
    vm_query.update({Resource.name: prefix + Resource.name}, synchronize_session=False)
    session.commit()

def remove_prefix_in_individual_dag_permissions(session):
    if False:
        while True:
            i = 10
    dag_perms = ['can_read', 'can_edit']
    prefix = 'DAG:'
    perms = session.query(Permission).join(Action).filter(Action.name.in_(dag_perms)).join(Resource).filter(Resource.name.like(prefix + '%')).all()
    for permission in perms:
        permission.resource.name = permission.resource.name[len(prefix):]
    session.commit()

def get_or_create_dag_resource(session):
    if False:
        for i in range(10):
            print('nop')
    dag_resource = get_resource_query(session, permissions.RESOURCE_DAG).first()
    if dag_resource:
        return dag_resource
    dag_resource = Resource()
    dag_resource.name = permissions.RESOURCE_DAG
    session.add(dag_resource)
    session.commit()
    return dag_resource

def get_or_create_all_dag_resource(session):
    if False:
        return 10
    all_dag_resource = get_resource_query(session, 'all_dags').first()
    if all_dag_resource:
        return all_dag_resource
    all_dag_resource = Resource()
    all_dag_resource.name = 'all_dags'
    session.add(all_dag_resource)
    session.commit()
    return all_dag_resource

def get_or_create_action(session, action_name):
    if False:
        while True:
            i = 10
    action = get_action_query(session, action_name).first()
    if action:
        return action
    action = Action()
    action.name = action_name
    session.add(action)
    session.commit()
    return action

def get_resource_query(session, resource_name):
    if False:
        for i in range(10):
            print('nop')
    return session.query(Resource).filter(Resource.name == resource_name)

def get_action_query(session, action_name):
    if False:
        i = 10
        return i + 15
    return session.query(Action).filter(Action.name == action_name)

def get_permission_with_action_query(session, action):
    if False:
        i = 10
        return i + 15
    return session.query(Permission).filter(Permission.action == action)

def get_permission_with_resource_query(session, resource):
    if False:
        i = 10
        return i + 15
    return session.query(Permission).filter(Permission.resource_id == resource.id)

def update_permission_action(session, permission_query, action):
    if False:
        i = 10
        return i + 15
    permission_query.update({Permission.action_id: action.id}, synchronize_session=False)
    session.commit()

def get_permission(session, resource, action):
    if False:
        while True:
            i = 10
    return session.query(Permission).filter(Permission.resource == resource).filter(Permission.action == action).first()

def update_permission_resource(session, permission_query, resource):
    if False:
        while True:
            i = 10
    for permission in permission_query.all():
        if not get_permission(session, resource, permission.action):
            permission.resource = resource
        else:
            session.delete(permission)
    session.commit()

def migrate_to_new_dag_permissions(db):
    if False:
        print('Hello World!')
    prefix_individual_dag_permissions(db.session)
    can_dag_read_action = get_action_query(db.session, 'can_dag_read').first()
    old_can_dag_read_permissions = get_permission_with_action_query(db.session, can_dag_read_action)
    can_read_action = get_or_create_action(db.session, 'can_read')
    update_permission_action(db.session, old_can_dag_read_permissions, can_read_action)
    can_dag_edit_action = get_action_query(db.session, 'can_dag_edit').first()
    old_can_dag_edit_permissions = get_permission_with_action_query(db.session, can_dag_edit_action)
    can_edit_action = get_or_create_action(db.session, 'can_edit')
    update_permission_action(db.session, old_can_dag_edit_permissions, can_edit_action)
    all_dags_resource = get_resource_query(db.session, 'all_dags').first()
    if all_dags_resource:
        old_all_dags_permission = get_permission_with_resource_query(db.session, all_dags_resource)
        dag_resource = get_or_create_dag_resource(db.session)
        update_permission_resource(db.session, old_all_dags_permission, dag_resource)
        db.session.delete(all_dags_resource)
    if can_dag_read_action:
        db.session.delete(can_dag_read_action)
    if can_dag_edit_action:
        db.session.delete(can_dag_edit_action)
    db.session.commit()

def undo_migrate_to_new_dag_permissions(session):
    if False:
        return 10
    remove_prefix_in_individual_dag_permissions(session)
    can_read_action = get_action_query(session, 'can_read').first()
    new_can_read_permissions = get_permission_with_action_query(session, can_read_action)
    can_dag_read_action = get_or_create_action(session, 'can_dag_read')
    update_permission_action(session, new_can_read_permissions, can_dag_read_action)
    can_edit_action = get_action_query(session, 'can_edit').first()
    new_can_edit_permissions = get_permission_with_action_query(session, can_edit_action)
    can_dag_edit_action = get_or_create_action(session, 'can_dag_edit')
    update_permission_action(session, new_can_edit_permissions, can_dag_edit_action)
    dag_resource = get_resource_query(session, permissions.RESOURCE_DAG).first()
    if dag_resource:
        new_dag_permission = get_permission_with_resource_query(session, dag_resource)
        old_all_dag_resource = get_or_create_all_dag_resource(session)
        update_permission_resource(session, new_dag_permission, old_all_dag_resource)
        session.delete(dag_resource)
    if can_read_action:
        session.delete(can_read_action)
    if can_edit_action:
        session.delete(can_edit_action)
    session.commit()

def upgrade():
    if False:
        return 10
    db = SQLA()
    db.session = settings.Session
    migrate_to_new_dag_permissions(db)
    db.session.commit()
    db.session.close()

def downgrade():
    if False:
        print('Hello World!')
    db = SQLA()
    db.session = settings.Session
    undo_migrate_to_new_dag_permissions(db.session)