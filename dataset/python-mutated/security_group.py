import logging
from pprint import pp
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class SecurityGroupWrapper:
    """Encapsulates Amazon Elastic Compute Cloud (Amazon EC2) security group actions."""

    def __init__(self, ec2_resource, security_group=None):
        if False:
            while True:
                i = 10
        '\n        :param ec2_resource: A Boto3 Amazon EC2 resource. This high-level resource\n                             is used to create additional high-level objects\n                             that wrap low-level Amazon EC2 service actions.\n        :param security_group: A Boto3 SecurityGroup object. This is a high-level object\n                               that wraps security group actions.\n        '
        self.ec2_resource = ec2_resource
        self.security_group = security_group

    @classmethod
    def from_resource(cls):
        if False:
            for i in range(10):
                print('nop')
        ec2_resource = boto3.resource('ec2')
        return cls(ec2_resource)

    def create(self, group_name, group_description):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a security group in the default virtual private cloud (VPC) of the\n        current account.\n\n        :param group_name: The name of the security group to create.\n        :param group_description: The description of the security group to create.\n        :return: A Boto3 SecurityGroup object that represents the newly created security group.\n        '
        try:
            self.security_group = self.ec2_resource.create_security_group(GroupName=group_name, Description=group_description)
        except ClientError as err:
            logger.error("Couldn't create security group %s. Here's why: %s: %s", group_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return self.security_group

    def authorize_ingress(self, ssh_ingress_ip):
        if False:
            print('Hello World!')
        "\n        Adds a rule to the security group to allow access to SSH.\n\n        :param ssh_ingress_ip: The IP address that is granted inbound access to connect\n                               to port 22 over TCP, used for SSH.\n        :return: The response to the authorization request. The 'Return' field of the\n                 response indicates whether the request succeeded or failed.\n        "
        if self.security_group is None:
            logger.info('No security group to update.')
            return
        try:
            ip_permissions = [{'IpProtocol': 'tcp', 'FromPort': 22, 'ToPort': 22, 'IpRanges': [{'CidrIp': f'{ssh_ingress_ip}/32'}]}]
            response = self.security_group.authorize_ingress(IpPermissions=ip_permissions)
        except ClientError as err:
            logger.error("Couldn't authorize inbound rules for %s. Here's why: %s: %s", self.security_group.id, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return response

    def describe(self):
        if False:
            i = 10
            return i + 15
        '\n        Displays information about the security group.\n        '
        if self.security_group is None:
            logger.info('No security group to describe.')
            return
        try:
            print(f'Security group: {self.security_group.group_name}')
            print(f'\tID: {self.security_group.id}')
            print(f'\tVPC: {self.security_group.vpc_id}')
            if self.security_group.ip_permissions:
                print(f'Inbound permissions:')
                pp(self.security_group.ip_permissions)
        except ClientError as err:
            logger.error("Couldn't get data for security group %s. Here's why: %s: %s", self.security_group.id, err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def delete(self):
        if False:
            return 10
        '\n        Deletes the security group.\n        '
        if self.security_group is None:
            logger.info('No security group to delete.')
            return
        group_id = self.security_group.id
        try:
            self.security_group.delete()
        except ClientError as err:
            logger.error("Couldn't delete security group %s. Here's why: %s: %s", group_id, err.response['Error']['Code'], err.response['Error']['Message'])
            raise