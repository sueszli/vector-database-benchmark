"""
Purpose

Shows how to use AWS Identity and Access Management (IAM) policies.
"""
import json
import logging
import operator
import pprint
import time
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)
iam = boto3.resource('iam')

def create_policy(name, description, actions, resource_arn):
    if False:
        print('Hello World!')
    "\n    Creates a policy that contains a single statement.\n\n    :param name: The name of the policy to create.\n    :param description: The description of the policy.\n    :param actions: The actions allowed by the policy. These typically take the\n                    form of service:action, such as s3:PutObject.\n    :param resource_arn: The Amazon Resource Name (ARN) of the resource this policy\n                         applies to. This ARN can contain wildcards, such as\n                         'arn:aws:s3:::my-bucket/*' to allow actions on all objects\n                         in the bucket named 'my-bucket'.\n    :return: The newly created policy.\n    "
    policy_doc = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': actions, 'Resource': resource_arn}]}
    try:
        policy = iam.create_policy(PolicyName=name, Description=description, PolicyDocument=json.dumps(policy_doc))
        logger.info('Created policy %s.', policy.arn)
    except ClientError:
        logger.exception("Couldn't create policy %s.", name)
        raise
    else:
        return policy

def delete_policy(policy_arn):
    if False:
        while True:
            i = 10
    '\n    Deletes a policy.\n\n    :param policy_arn: The ARN of the policy to delete.\n    '
    try:
        iam.Policy(policy_arn).delete()
        logger.info('Deleted policy %s.', policy_arn)
    except ClientError:
        logger.exception("Couldn't delete policy %s.", policy_arn)
        raise

def create_policy_version(policy_arn, actions, resource_arn, set_as_default):
    if False:
        while True:
            i = 10
    '\n    Creates a policy version. Policies can have up to five versions. The default\n    version is the one that is used for all resources that reference the policy.\n\n    :param policy_arn: The ARN of the policy.\n    :param actions: The actions to allow in the policy version.\n    :param resource_arn: The ARN of the resource this policy version applies to.\n    :param set_as_default: When True, this policy version is set as the default\n                           version for the policy. Otherwise, the default\n                           is not changed.\n    :return: The newly created policy version.\n    '
    policy_doc = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': actions, 'Resource': resource_arn}]}
    try:
        policy = iam.Policy(policy_arn)
        policy_version = policy.create_version(PolicyDocument=json.dumps(policy_doc), SetAsDefault=set_as_default)
        logger.info('Created policy version %s for policy %s.', policy_version.version_id, policy_version.arn)
    except ClientError:
        logger.exception("Couldn't create a policy version for %s.", policy_arn)
        raise
    else:
        return policy_version

def list_policies(scope):
    if False:
        print('Hello World!')
    "\n    Lists the policies in the current account.\n\n    :param scope: Limits the kinds of policies that are returned. For example,\n                  'Local' specifies that only locally managed policies are returned.\n    :return: The list of policies.\n    "
    try:
        policies = list(iam.policies.filter(Scope=scope))
        logger.info("Got %s policies in scope '%s'.", len(policies), scope)
    except ClientError:
        logger.exception("Couldn't get policies for scope '%s'.", scope)
        raise
    else:
        return policies

def get_default_policy_statement(policy_arn):
    if False:
        return 10
    '\n    Gets the statement of the default version of the specified policy.\n\n    :param policy_arn: The ARN of the policy to look up.\n    :return: The statement of the default policy version.\n    '
    try:
        policy = iam.Policy(policy_arn)
        policy_doc = policy.default_version.document
        policy_statement = policy_doc.get('Statement', None)
        logger.info('Got default policy doc for %s.', policy.policy_name)
        logger.info(policy_doc)
    except ClientError:
        logger.exception("Couldn't get default policy statement for %s.", policy_arn)
        raise
    else:
        return policy_statement

def rollback_policy_version(policy_arn):
    if False:
        while True:
            i = 10
    '\n    Rolls back to the previous default policy, if it exists.\n\n    1. Gets the list of policy versions in order by date.\n    2. Finds the default.\n    3. Makes the previous policy the default.\n    4. Deletes the old default version.\n\n    :param policy_arn: The ARN of the policy to roll back.\n    :return: The default version of the policy after the rollback.\n    '
    try:
        policy_versions = sorted(iam.Policy(policy_arn).versions.all(), key=operator.attrgetter('create_date'))
        logger.info('Got %s versions for %s.', len(policy_versions), policy_arn)
    except ClientError:
        logger.exception("Couldn't get versions for %s.", policy_arn)
        raise
    default_version = None
    rollback_version = None
    try:
        while default_version is None:
            ver = policy_versions.pop()
            if ver.is_default_version:
                default_version = ver
        rollback_version = policy_versions.pop()
        rollback_version.set_as_default()
        logger.info('Set %s as the default version.', rollback_version.version_id)
        default_version.delete()
        logger.info('Deleted original default version %s.', default_version.version_id)
    except IndexError:
        if default_version is None:
            logger.warning('No default version found for %s.', policy_arn)
        elif rollback_version is None:
            logger.warning('Default version %s found for %s, but no previous version exists, so nothing to roll back to.', default_version.version_id, policy_arn)
    except ClientError:
        logger.exception("Couldn't roll back version for %s.", policy_arn)
        raise
    else:
        return rollback_version

def attach_to_role(role_name, policy_arn):
    if False:
        print('Hello World!')
    '\n    Attaches a policy to a role.\n\n    :param role_name: The name of the role. **Note** this is the name, not the ARN.\n    :param policy_arn: The ARN of the policy.\n    '
    try:
        iam.Policy(policy_arn).attach_role(RoleName=role_name)
        logger.info('Attached policy %s to role %s.', policy_arn, role_name)
    except ClientError:
        logger.exception("Couldn't attach policy %s to role %s.", policy_arn, role_name)
        raise

def detach_from_role(role_name, policy_arn):
    if False:
        for i in range(10):
            print('nop')
    '\n    Detaches a policy from a role.\n\n    :param role_name: The name of the role. **Note** this is the name, not the ARN.\n    :param policy_arn: The ARN of the policy.\n    '
    try:
        iam.Policy(policy_arn).detach_role(RoleName=role_name)
        logger.info('Detached policy %s from role %s.', policy_arn, role_name)
    except ClientError:
        logger.exception("Couldn't detach policy %s from role %s.", policy_arn, role_name)
        raise

def usage_demo():
    if False:
        return 10
    'Shows how to use the policy functions.'
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print('-' * 88)
    print('Welcome to the AWS Identity and Account Management policy demo.')
    print('-' * 88)
    print('Policies let you define sets of permissions that can be attached to other IAM resources, like users and roles.')
    bucket_arn = f'arn:aws:s3:::made-up-bucket-name'
    policy = create_policy('demo-iam-policy', 'Policy for IAM demonstration.', ['s3:ListObjects'], bucket_arn)
    print(f'Created policy {policy.policy_name}.')
    policies = list_policies('Local')
    print(f'Your account has {len(policies)} managed policies:')
    print(*[pol.policy_name for pol in policies], sep=', ')
    time.sleep(1)
    policy_version = create_policy_version(policy.arn, ['s3:PutObject'], bucket_arn, True)
    print(f'Added policy version {policy_version.version_id} to policy {policy.policy_name}.')
    default_statement = get_default_policy_statement(policy.arn)
    print(f'The default policy statement for {policy.policy_name} is:')
    pprint.pprint(default_statement)
    rollback_version = rollback_policy_version(policy.arn)
    print(f'Rolled back to version {rollback_version.version_id} for {policy.policy_name}.')
    default_statement = get_default_policy_statement(policy.arn)
    print(f'The default policy statement for {policy.policy_name} is now:')
    pprint.pprint(default_statement)
    delete_policy(policy.arn)
    print(f'Deleted policy {policy.policy_name}.')
    print('Thanks for watching!')
if __name__ == '__main__':
    usage_demo()