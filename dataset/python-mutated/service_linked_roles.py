"""
Purpose

Shows how to use AWS Identity and Access Management (IAM) service-linked roles.
"""
import logging
from pprint import pprint
import time
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)
iam = boto3.resource('iam')

def create_service_linked_role(service_name, description):
    if False:
        print('Hello World!')
    '\n    Creates a service-linked role.\n\n    :param service_name: The name of the service that owns the role.\n    :param description: A description to give the role.\n    :return: The newly created role.\n    '
    try:
        response = iam.meta.client.create_service_linked_role(AWSServiceName=service_name, Description=description)
        role = iam.Role(response['Role']['RoleName'])
        logger.info('Created service-linked role %s.', role.name)
    except ClientError:
        logger.exception("Couldn't create service-linked role for %s.", service_name)
        raise
    else:
        return role

def usage_demo():
    if False:
        while True:
            i = 10
    print('-' * 88)
    print('Welcome to the IAM service-linked role demo!')
    print('-' * 88)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    service_name = input("Enter the name of a service to create a service-linked role.\nFor example, 'elasticbeanstalk.amazonaws.com' or 'batch.amazonaws.com': ")
    role = create_service_linked_role(service_name, 'Service-linked role demo.')
    policy = list(role.attached_policies.all())[0]
    print(f'The policy document for {role.name} is:')
    pprint(policy.default_version.document)
    if role is not None:
        answer = input('Do you want to delete the role? You should only do this if you are sure it is not being used. (y/n)? ')
        if answer.lower() == 'y':
            try:
                response = iam.meta.client.delete_service_linked_role(RoleName=role.name)
                task_id = response['DeletionTaskId']
                while True:
                    response = iam.meta.client.get_service_linked_role_deletion_status(DeletionTaskId=task_id)
                    status = response['Status']
                    print(f'Deletion of {role.name} {status}.')
                    if status in ['SUCCEEDED', 'FAILED']:
                        break
                    else:
                        time.sleep(3)
            except ClientError as error:
                if error.response['Error']['Code'] == 'NoSuchEntity':
                    pass
                else:
                    print(f"Couldn't delete {role.name}. Here's why: {error.response['Error']['Code']}")
                    raise
    print('Thanks for watching!')
    print('-' * 88)
if __name__ == '__main__':
    try:
        usage_demo()
    except Exception as err:
        print('Something went wrong with the demo:')
        print(err)