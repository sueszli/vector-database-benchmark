import logging
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class ElasticIpWrapper:
    """Encapsulates Amazon Elastic Compute Cloud (Amazon EC2) Elastic IP address actions."""

    def __init__(self, ec2_resource, elastic_ip=None):
        if False:
            print('Hello World!')
        '\n        :param ec2_resource: A Boto3 Amazon EC2 resource. This high-level resource\n                             is used to create additional high-level objects\n                             that wrap low-level Amazon EC2 service actions.\n        :param elastic_ip: A Boto3 VpcAddress object. This is a high-level object that\n                           wraps Elastic IP actions.\n        '
        self.ec2_resource = ec2_resource
        self.elastic_ip = elastic_ip

    @classmethod
    def from_resource(cls):
        if False:
            while True:
                i = 10
        ec2_resource = boto3.resource('ec2')
        return cls(ec2_resource)

    def allocate(self):
        if False:
            print('Hello World!')
        '\n        Allocates an Elastic IP address that can be associated with an Amazon EC2\n        instance. By using an Elastic IP address, you can keep the public IP address\n        constant even when you restart the associated instance.\n\n        :return: The newly created Elastic IP object. By default, the address is not\n                 associated with any instance.\n        '
        try:
            response = self.ec2_resource.meta.client.allocate_address(Domain='vpc')
            self.elastic_ip = self.ec2_resource.VpcAddress(response['AllocationId'])
        except ClientError as err:
            logger.error("Couldn't allocate Elastic IP. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return self.elastic_ip

    def associate(self, instance):
        if False:
            return 10
        "\n        Associates an Elastic IP address with an instance. When this association is\n        created, the Elastic IP's public IP address is immediately used as the public\n        IP address of the associated instance.\n\n        :param instance: A Boto3 Instance object. This is a high-level object that wraps\n                         Amazon EC2 instance actions.\n        :return: A response that contains the ID of the association.\n        "
        if self.elastic_ip is None:
            logger.info('No Elastic IP to associate.')
            return
        try:
            response = self.elastic_ip.associate(InstanceId=instance.id)
        except ClientError as err:
            logger.error("Couldn't associate Elastic IP %s with instance %s. Here's why: %s: %s", self.elastic_ip.allocation_id, instance.id, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        return response

    def disassociate(self):
        if False:
            i = 10
            return i + 15
        '\n        Removes an association between an Elastic IP address and an instance. When the\n        association is removed, the instance is assigned a new public IP address.\n        '
        if self.elastic_ip is None:
            logger.info('No Elastic IP to disassociate.')
            return
        try:
            self.elastic_ip.association.delete()
        except ClientError as err:
            logger.error("Couldn't disassociate Elastic IP %s from its instance. Here's why: %s: %s", self.elastic_ip.allocation_id, err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def release(self):
        if False:
            i = 10
            return i + 15
        '\n        Releases an Elastic IP address. After the Elastic IP address is released,\n        it can no longer be used.\n        '
        if self.elastic_ip is None:
            logger.info('No Elastic IP to release.')
            return
        try:
            self.elastic_ip.release()
        except ClientError as err:
            logger.error("Couldn't release Elastic IP address %s. Here's why: %s: %s", self.elastic_ip.allocation_id, err.response['Error']['Code'], err.response['Error']['Message'])
            raise