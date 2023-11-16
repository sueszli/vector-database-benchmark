import logging
import os
import tempfile
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class KeyPairWrapper:
    """Encapsulates Amazon Elastic Compute Cloud (Amazon EC2) key pair actions."""

    def __init__(self, ec2_resource, key_file_dir, key_pair=None):
        if False:
            return 10
        '\n        :param ec2_resource: A Boto3 Amazon EC2 resource. This high-level resource\n                             is used to create additional high-level objects\n                             that wrap low-level Amazon EC2 service actions.\n        :param key_file_dir: The folder where the private key information is stored.\n                             This should be a secure folder.\n        :param key_pair: A Boto3 KeyPair object. This is a high-level object that\n                         wraps key pair actions.\n        '
        self.ec2_resource = ec2_resource
        self.key_pair = key_pair
        self.key_file_path = None
        self.key_file_dir = key_file_dir

    @classmethod
    def from_resource(cls):
        if False:
            print('Hello World!')
        ec2_resource = boto3.resource('ec2')
        return cls(ec2_resource, tempfile.TemporaryDirectory())

    def create(self, key_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a key pair that can be used to securely connect to an EC2 instance.\n        The returned key pair contains private key information that cannot be retrieved\n        again. The private key data is stored as a .pem file.\n\n        :param key_name: The name of the key pair to create.\n        :return: A Boto3 KeyPair object that represents the newly created key pair.\n        '
        try:
            self.key_pair = self.ec2_resource.create_key_pair(KeyName=key_name)
            self.key_file_path = os.path.join(self.key_file_dir.name, f'{self.key_pair.name}.pem')
            with open(self.key_file_path, 'w') as key_file:
                key_file.write(self.key_pair.key_material)
        except ClientError as err:
            logger.error("Couldn't create key %s. Here's why: %s: %s", key_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise
        else:
            return self.key_pair

    def list(self, limit):
        if False:
            i = 10
            return i + 15
        '\n        Displays a list of key pairs for the current account.\n\n        :param limit: The maximum number of key pairs to list.\n        '
        try:
            for kp in self.ec2_resource.key_pairs.limit(limit):
                print(f'Found {kp.key_type} key {kp.name} with fingerprint:')
                print(f'\t{kp.key_fingerprint}')
        except ClientError as err:
            logger.error("Couldn't list key pairs. Here's why: %s: %s", err.response['Error']['Code'], err.response['Error']['Message'])
            raise

    def delete(self):
        if False:
            return 10
        '\n        Deletes a key pair.\n        '
        if self.key_pair is None:
            logger.info('No key pair to delete.')
            return
        key_name = self.key_pair.name
        try:
            self.key_pair.delete()
            self.key_pair = None
        except ClientError as err:
            logger.error("Couldn't delete key %s. Here's why: %s : %s", key_name, err.response['Error']['Code'], err.response['Error']['Message'])
            raise