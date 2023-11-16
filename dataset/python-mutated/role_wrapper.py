"""
Purpose

Shows how to use AWS Identity and Access Management (IAM) roles.
"""
import json
import logging
import pprint
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)
iam = boto3.resource('iam')

def create_role(role_name, allowed_services):
    if False:
        i = 10
        return i + 15
    '\n    Creates a role that lets a list of specified services assume the role.\n\n    :param role_name: The name of the role.\n    :param allowed_services: The services that can assume the role.\n    :return: The newly created role.\n    '
    trust_policy = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Principal': {'Service': service}, 'Action': 'sts:AssumeRole'} for service in allowed_services]}
    try:
        role = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(trust_policy))
        logger.info('Created role %s.', role.name)
    except ClientError:
        logger.exception("Couldn't create role %s.", role_name)
        raise
    else:
        return role

def get_role(role_name):
    if False:
        print('Hello World!')
    '\n    Gets a role by name.\n\n    :param role_name: The name of the role to retrieve.\n    :return: The specified role.\n    '
    try:
        role = iam.Role(role_name)
        role.load()
        logger.info('Got role with arn %s.', role.arn)
    except ClientError:
        logger.exception("Couldn't get role named %s.", role_name)
        raise
    else:
        return role

def list_roles(count):
    if False:
        for i in range(10):
            print('nop')
    '\n    Lists the specified number of roles for the account.\n\n    :param count: The number of roles to list.\n    '
    try:
        roles = list(iam.roles.limit(count=count))
        for role in roles:
            logger.info('Role: %s', role.name)
    except ClientError:
        logger.exception("Couldn't list roles for the account.")
        raise
    else:
        return roles

def delete_role(role_name):
    if False:
        for i in range(10):
            print('nop')
    '\n    Deletes a role.\n\n    :param role_name: The name of the role to delete.\n    '
    try:
        iam.Role(role_name).delete()
        logger.info('Deleted role %s.', role_name)
    except ClientError:
        logger.exception("Couldn't delete role %s.", role_name)
        raise

def attach_policy(role_name, policy_arn):
    if False:
        for i in range(10):
            print('nop')
    '\n    Attaches a policy to a role.\n\n    :param role_name: The name of the role. **Note** this is the name, not the ARN.\n    :param policy_arn: The ARN of the policy.\n    '
    try:
        iam.Role(role_name).attach_policy(PolicyArn=policy_arn)
        logger.info('Attached policy %s to role %s.', policy_arn, role_name)
    except ClientError:
        logger.exception("Couldn't attach policy %s to role %s.", policy_arn, role_name)
        raise

def list_policies(role_name):
    if False:
        while True:
            i = 10
    '\n    Lists inline policies for a role.\n\n    :param role_name: The name of the role to query.\n    '
    try:
        role = iam.Role(role_name)
        for policy in role.policies.all():
            logger.info('Got inline policy %s.', policy.name)
    except ClientError:
        logger.exception("Couldn't list inline policies for %s.", role_name)
        raise

def list_attached_policies(role_name):
    if False:
        i = 10
        return i + 15
    '\n    Lists policies attached to a role.\n\n    :param role_name: The name of the role to query.\n    '
    try:
        role = iam.Role(role_name)
        for policy in role.attached_policies.all():
            logger.info('Got policy %s.', policy.arn)
    except ClientError:
        logger.exception("Couldn't list attached policies for %s.", role_name)
        raise

def detach_policy(role_name, policy_arn):
    if False:
        return 10
    '\n    Detaches a policy from a role.\n\n    :param role_name: The name of the role. **Note** this is the name, not the ARN.\n    :param policy_arn: The ARN of the policy.\n    '
    try:
        iam.Role(role_name).detach_policy(PolicyArn=policy_arn)
        logger.info('Detached policy %s from role %s.', policy_arn, role_name)
    except ClientError:
        logger.exception("Couldn't detach policy %s from role %s.", policy_arn, role_name)
        raise

def usage_demo():
    if False:
        i = 10
        return i + 15
    'Shows how to use the role functions.'
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print('-' * 88)
    print('Welcome to the AWS Identity and Account Management role demo.')
    print('-' * 88)
    print('Roles let you define sets of permissions and can be assumed by other entities, like users and services.')
    print('The first 10 roles currently in your account are:')
    roles = list_roles(10)
    print(f'The inline policies for role {roles[0].name} are:')
    list_policies(roles[0].name)
    role = create_role('demo-iam-role', ['lambda.amazonaws.com', 'batchoperations.s3.amazonaws.com'])
    print(f'Created role {role.name}, with trust policy:')
    pprint.pprint(role.assume_role_policy_document)
    policy_arn = 'arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess'
    attach_policy(role.name, policy_arn)
    print(f'Attached policy {policy_arn} to {role.name}.')
    print(f'Policies attached to role {role.name} are:')
    list_attached_policies(role.name)
    detach_policy(role.name, policy_arn)
    print(f'Detached policy {policy_arn} from {role.name}.')
    delete_role(role.name)
    print(f'Deleted {role.name}.')
    print('Thanks for watching!')
if __name__ == '__main__':
    usage_demo()