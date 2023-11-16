"""Demonstrates how to perform basic operations with Google Cloud IAM
custom roles.

For more information, see the documentation at
https://cloud.google.com/iam/docs/creating-custom-roles.
"""
import argparse
import os
from google.oauth2 import service_account
import googleapiclient.discovery
credentials = service_account.Credentials.from_service_account_file(filename=os.environ['GOOGLE_APPLICATION_CREDENTIALS'], scopes=['https://www.googleapis.com/auth/cloud-platform'])
service = googleapiclient.discovery.build('iam', 'v1', credentials=credentials)

def query_testable_permissions(resource: str) -> None:
    if False:
        print('Hello World!')
    'Lists valid permissions for a resource.'
    permissions = service.permissions().queryTestablePermissions(body={'fullResourceName': resource}).execute()['permissions']
    for p in permissions:
        print(p['name'])

def get_role(name: str) -> None:
    if False:
        return 10
    'Gets a role.'
    role = service.roles().get(name=name).execute()
    print(role['name'])
    for permission in role['includedPermissions']:
        print(permission)

def create_role(name: str, project: str, title: str, description: str, permissions: str, stage: str) -> dict:
    if False:
        while True:
            i = 10
    'Creates a role.'
    role = service.projects().roles().create(parent='projects/' + project, body={'roleId': name, 'role': {'title': title, 'description': description, 'includedPermissions': permissions, 'stage': stage}}).execute()
    print('Created role: ' + role['name'])
    return role

def edit_role(name: str, project: str, title: str, description: str, permissions: str, stage: str) -> dict:
    if False:
        return 10
    'Creates a role.'
    role = service.projects().roles().patch(name='projects/' + project + '/roles/' + name, body={'title': title, 'description': description, 'includedPermissions': permissions, 'stage': stage}).execute()
    print('Updated role: ' + role['name'])
    return role

def list_roles(project_id: str) -> None:
    if False:
        print('Hello World!')
    'Lists roles.'
    roles = service.roles().list(parent='projects/' + project_id).execute()['roles']
    for role in roles:
        print(role['name'])

def disable_role(name: str, project: str) -> dict:
    if False:
        return 10
    'Disables a role.'
    role = service.projects().roles().patch(name='projects/' + project + '/roles/' + name, body={'stage': 'DISABLED'}).execute()
    print('Disabled role: ' + role['name'])
    return role

def delete_role(name: str, project: str) -> dict:
    if False:
        return 10
    'Deletes a role.'
    role = service.projects().roles().delete(name='projects/' + project + '/roles/' + name).execute()
    print('Deleted role: ' + name)
    return role

def undelete_role(name: str, project: str) -> dict:
    if False:
        return 10
    'Undeletes a role.'
    role = service.projects().roles().patch(name='projects/' + project + '/roles/' + name, body={'stage': 'DISABLED'}).execute()
    print('Disabled role: ' + role['name'])
    return role

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    view_permissions_parser = subparsers.add_parser('permissions', help=query_testable_permissions.__doc__)
    view_permissions_parser.add_argument('resource')
    get_role_parser = subparsers.add_parser('get', help=get_role.__doc__)
    get_role_parser.add_argument('name')
    get_role_parser = subparsers.add_parser('create', help=create_role.__doc__)
    get_role_parser.add_argument('name')
    get_role_parser.add_argument('project')
    get_role_parser.add_argument('title')
    get_role_parser.add_argument('description')
    get_role_parser.add_argument('permissions')
    get_role_parser.add_argument('stage')
    edit_role_parser = subparsers.add_parser('edit', help=create_role.__doc__)
    edit_role_parser.add_argument('name')
    edit_role_parser.add_argument('project')
    edit_role_parser.add_argument('title')
    edit_role_parser.add_argument('description')
    edit_role_parser.add_argument('permissions')
    edit_role_parser.add_argument('stage')
    list_roles_parser = subparsers.add_parser('list', help=list_roles.__doc__)
    list_roles_parser.add_argument('project_id')
    disable_role_parser = subparsers.add_parser('disable', help=get_role.__doc__)
    disable_role_parser.add_argument('name')
    disable_role_parser.add_argument('project')
    delete_role_parser = subparsers.add_parser('delete', help=get_role.__doc__)
    delete_role_parser.add_argument('name')
    delete_role_parser.add_argument('project')
    undelete_role_parser = subparsers.add_parser('undelete', help=get_role.__doc__)
    undelete_role_parser.add_argument('name')
    undelete_role_parser.add_argument('project')
    args = parser.parse_args()
    if args.command == 'permissions':
        query_testable_permissions(args.resource)
    elif args.command == 'get':
        get_role(args.name)
    elif args.command == 'list':
        list_roles(args.project_id)
    elif args.command == 'create':
        create_role(args.name, args.project, args.title, args.description, args.permissions, args.stage)
    elif args.command == 'edit':
        edit_role(args.name, args.project, args.title, args.description, args.permissions, args.stage)
    elif args.command == 'disable':
        disable_role(args.name, args.project)
    elif args.command == 'delete':
        delete_role(args.name, args.project)
    elif args.command == 'undelete':
        undelete_role(args.name, args.project)
if __name__ == '__main__':
    main()