"""Demonstrates how to perform basic access management with Google Cloud IAM.

For more information, see the documentation at
https://cloud.google.com/iam/docs/granting-changing-revoking-access.
"""
import argparse
import os
from google.oauth2 import service_account
import googleapiclient.discovery

def get_policy(project_id: str, version: int=1) -> dict:
    if False:
        for i in range(10):
            print('nop')
    'Gets IAM policy for a project.'
    credentials = service_account.Credentials.from_service_account_file(filename=os.environ['GOOGLE_APPLICATION_CREDENTIALS'], scopes=['https://www.googleapis.com/auth/cloud-platform'])
    service = googleapiclient.discovery.build('cloudresourcemanager', 'v1', credentials=credentials)
    policy = service.projects().getIamPolicy(resource=project_id, body={'options': {'requestedPolicyVersion': version}}).execute()
    print(policy)
    return policy

def modify_policy_add_member(policy: dict, role: str, member: str) -> dict:
    if False:
        print('Hello World!')
    'Adds a new member to a role binding.'
    binding = next((b for b in policy['bindings'] if b['role'] == role))
    binding['members'].append(member)
    print(binding)
    return policy

def modify_policy_add_role(policy: dict, role: str, member: str) -> dict:
    if False:
        return 10
    'Adds a new role binding to a policy.'
    binding = {'role': role, 'members': [member]}
    policy['bindings'].append(binding)
    print(policy)
    return policy

def modify_policy_remove_member(policy: dict, role: str, member: str) -> dict:
    if False:
        print('Hello World!')
    'Removes a  member from a role binding.'
    binding = next((b for b in policy['bindings'] if b['role'] == role))
    if 'members' in binding and member in binding['members']:
        binding['members'].remove(member)
    print(binding)
    return policy

def set_policy(project_id: str, policy: dict) -> dict:
    if False:
        return 10
    'Sets IAM policy for a project.'
    credentials = service_account.Credentials.from_service_account_file(filename=os.environ['GOOGLE_APPLICATION_CREDENTIALS'], scopes=['https://www.googleapis.com/auth/cloud-platform'])
    service = googleapiclient.discovery.build('cloudresourcemanager', 'v1', credentials=credentials)
    policy = service.projects().setIamPolicy(resource=project_id, body={'policy': policy}).execute()
    print(policy)
    return policy

def test_permissions(project_id: str) -> dict:
    if False:
        return 10
    'Tests IAM permissions of the caller'
    credentials = service_account.Credentials.from_service_account_file(filename=os.environ['GOOGLE_APPLICATION_CREDENTIALS'], scopes=['https://www.googleapis.com/auth/cloud-platform'])
    service = googleapiclient.discovery.build('cloudresourcemanager', 'v1', credentials=credentials)
    permissions = {'permissions': ['resourcemanager.projects.get', 'resourcemanager.projects.delete']}
    request = service.projects().testIamPermissions(resource=project_id, body=permissions)
    returnedPermissions = request.execute()
    print(returnedPermissions)
    return returnedPermissions

def main() -> None:
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command')
    get_parser = subparsers.add_parser('get', help=get_policy.__doc__)
    get_parser.add_argument('project_id')
    modify_member_parser = subparsers.add_parser('modify_member', help=get_policy.__doc__)
    modify_member_parser.add_argument('project_id')
    modify_member_parser.add_argument('role')
    modify_member_parser.add_argument('member')
    modify_role_parser = subparsers.add_parser('modify_role', help=get_policy.__doc__)
    modify_role_parser.add_argument('project_id')
    modify_role_parser.add_argument('project_id')
    modify_role_parser.add_argument('role')
    modify_role_parser.add_argument('member')
    modify_member_parser = subparsers.add_parser('modify_member', help=get_policy.__doc__)
    modify_member_parser.add_argument('project_id')
    modify_member_parser.add_argument('role')
    modify_member_parser.add_argument('member')
    set_parser = subparsers.add_parser('set', help=set_policy.__doc__)
    set_parser.add_argument('project_id')
    set_parser.add_argument('policy')
    test_permissions_parser = subparsers.add_parser('test_permissions', help=get_policy.__doc__)
    test_permissions_parser.add_argument('project_id')
    args = parser.parse_args()
    if args.command == 'get':
        get_policy(args.project_id)
    elif args.command == 'set':
        set_policy(args.project_id, args.policy)
    elif args.command == 'add_member':
        modify_policy_add_member(args.policy, args.role, args.member)
    elif args.command == 'remove_member':
        modify_policy_remove_member(args.policy, args.role, args.member)
    elif args.command == 'add_binding':
        modify_policy_add_role(args.policy, args.role, args.member)
    elif args.command == 'test_permissions':
        test_permissions(args.project_id)
if __name__ == '__main__':
    main()