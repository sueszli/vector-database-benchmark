"""
Purpose

Show how to use AWS SDK for Python (Boto3) with Amazon Simple Storage Service
(Amazon S3) to perform basic bucket operations.
"""
import json
import logging
import uuid
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class BucketWrapper:
    """Encapsulates S3 bucket actions."""

    def __init__(self, bucket):
        if False:
            print('Hello World!')
        '\n        :param bucket: A Boto3 Bucket resource. This is a high-level resource in Boto3\n                       that wraps bucket actions in a class-like structure.\n        '
        self.bucket = bucket
        self.name = bucket.name

    def create(self, region_override=None):
        if False:
            while True:
                i = 10
        '\n        Create an Amazon S3 bucket in the default Region for the account or in the\n        specified Region.\n\n        :param region_override: The Region in which to create the bucket. If this is\n                                not specified, the Region configured in your shared\n                                credentials is used.\n        '
        if region_override is not None:
            region = region_override
        else:
            region = self.bucket.meta.client.meta.region_name
        try:
            self.bucket.create(CreateBucketConfiguration={'LocationConstraint': region})
            self.bucket.wait_until_exists()
            logger.info("Created bucket '%s' in region=%s", self.bucket.name, region)
        except ClientError as error:
            logger.exception("Couldn't create bucket named '%s' in region=%s.", self.bucket.name, region)
            raise error

    def exists(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine whether the bucket exists and you have access to it.\n\n        :return: True when the bucket exists; otherwise, False.\n        '
        try:
            self.bucket.meta.client.head_bucket(Bucket=self.bucket.name)
            logger.info('Bucket %s exists.', self.bucket.name)
            exists = True
        except ClientError:
            logger.warning("Bucket %s doesn't exist or you don't have access to it.", self.bucket.name)
            exists = False
        return exists

    @staticmethod
    def list(s3_resource):
        if False:
            while True:
                i = 10
        '\n        Get the buckets in all Regions for the current account.\n\n        :param s3_resource: A Boto3 S3 resource. This is a high-level resource in Boto3\n                            that contains collections and factory methods to create\n                            other high-level S3 sub-resources.\n        :return: The list of buckets.\n        '
        try:
            buckets = list(s3_resource.buckets.all())
            logger.info('Got buckets: %s.', buckets)
        except ClientError:
            logger.exception("Couldn't get buckets.")
            raise
        else:
            return buckets

    def delete(self):
        if False:
            i = 10
            return i + 15
        '\n        Delete the bucket. The bucket must be empty or an error is raised.\n        '
        try:
            self.bucket.delete()
            self.bucket.wait_until_not_exists()
            logger.info('Bucket %s successfully deleted.', self.bucket.name)
        except ClientError:
            logger.exception("Couldn't delete bucket %s.", self.bucket.name)
            raise

    def grant_log_delivery_access(self):
        if False:
            i = 10
            return i + 15
        '\n        Grant the AWS Log Delivery group write access to the bucket so that\n        Amazon S3 can deliver access logs to the bucket. This is the only recommended\n        use of an S3 bucket ACL.\n        '
        try:
            acl = self.bucket.Acl()
            grants = acl.grants if acl.grants else []
            grants.append({'Grantee': {'Type': 'Group', 'URI': 'http://acs.amazonaws.com/groups/s3/LogDelivery'}, 'Permission': 'WRITE'})
            acl.put(AccessControlPolicy={'Grants': grants, 'Owner': acl.owner})
            logger.info("Granted log delivery access to bucket '%s'", self.bucket.name)
        except ClientError:
            logger.exception("Couldn't add ACL to bucket '%s'.", self.bucket.name)
            raise

    def get_acl(self):
        if False:
            print('Hello World!')
        '\n        Get the ACL of the bucket.\n\n        :return: The ACL of the bucket.\n        '
        try:
            acl = self.bucket.Acl()
            logger.info('Got ACL for bucket %s. Owner is %s.', self.bucket.name, acl.owner)
        except ClientError:
            logger.exception("Couldn't get ACL for bucket %s.", self.bucket.name)
            raise
        else:
            return acl

    def put_cors(self, cors_rules):
        if False:
            for i in range(10):
                print('nop')
        '\n        Apply CORS rules to the bucket. CORS rules specify the HTTP actions that are\n        allowed from other domains.\n\n        :param cors_rules: The CORS rules to apply.\n        '
        try:
            self.bucket.Cors().put(CORSConfiguration={'CORSRules': cors_rules})
            logger.info("Put CORS rules %s for bucket '%s'.", cors_rules, self.bucket.name)
        except ClientError:
            logger.exception("Couldn't put CORS rules for bucket %s.", self.bucket.name)
            raise

    def get_cors(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the CORS rules for the bucket.\n\n        :return The CORS rules for the specified bucket.\n        '
        try:
            cors = self.bucket.Cors()
            logger.info("Got CORS rules %s for bucket '%s'.", cors.cors_rules, self.bucket.name)
        except ClientError:
            logger.exception(("Couldn't get CORS for bucket %s.", self.bucket.name))
            raise
        else:
            return cors

    def delete_cors(self):
        if False:
            i = 10
            return i + 15
        '\n        Delete the CORS rules from the bucket.\n\n        :param bucket_name: The name of the bucket to update.\n        '
        try:
            self.bucket.Cors().delete()
            logger.info("Deleted CORS from bucket '%s'.", self.bucket.name)
        except ClientError:
            logger.exception("Couldn't delete CORS from bucket '%s'.", self.bucket.name)
            raise

    def put_policy(self, policy):
        if False:
            while True:
                i = 10
        "\n        Apply a security policy to the bucket. Policies control users' ability\n        to perform specific actions, such as listing the objects in the bucket.\n\n        :param policy: The policy to apply to the bucket.\n        "
        try:
            self.bucket.Policy().put(Policy=json.dumps(policy))
            logger.info("Put policy %s for bucket '%s'.", policy, self.bucket.name)
        except ClientError:
            logger.exception("Couldn't apply policy to bucket '%s'.", self.bucket.name)
            raise

    def get_policy(self):
        if False:
            while True:
                i = 10
        '\n        Get the security policy of the bucket.\n\n        :return: The security policy of the specified bucket, in JSON format.\n        '
        try:
            policy = self.bucket.Policy()
            logger.info("Got policy %s for bucket '%s'.", policy.policy, self.bucket.name)
        except ClientError:
            logger.exception("Couldn't get policy for bucket '%s'.", self.bucket.name)
            raise
        else:
            return json.loads(policy.policy)

    def delete_policy(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Delete the security policy from the bucket.\n        '
        try:
            self.bucket.Policy().delete()
            logger.info("Deleted policy for bucket '%s'.", self.bucket.name)
        except ClientError:
            logger.exception("Couldn't delete policy for bucket '%s'.", self.bucket.name)
            raise

    def put_lifecycle_configuration(self, lifecycle_rules):
        if False:
            print('Hello World!')
        '\n        Apply a lifecycle configuration to the bucket. The lifecycle configuration can\n        be used to archive or delete the objects in the bucket according to specified\n        parameters, such as a number of days.\n\n        :param lifecycle_rules: The lifecycle rules to apply.\n        '
        try:
            self.bucket.LifecycleConfiguration().put(LifecycleConfiguration={'Rules': lifecycle_rules})
            logger.info("Put lifecycle rules %s for bucket '%s'.", lifecycle_rules, self.bucket.name)
        except ClientError:
            logger.exception("Couldn't put lifecycle rules for bucket '%s'.", self.bucket.name)
            raise

    def get_lifecycle_configuration(self):
        if False:
            print('Hello World!')
        '\n        Get the lifecycle configuration of the bucket.\n\n        :return: The lifecycle rules of the specified bucket.\n        '
        try:
            config = self.bucket.LifecycleConfiguration()
            logger.info("Got lifecycle rules %s for bucket '%s'.", config.rules, self.bucket.name)
        except:
            logger.exception("Couldn't get lifecycle rules for bucket '%s'.", self.bucket.name)
            raise
        else:
            return config.rules

    def delete_lifecycle_configuration(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove the lifecycle configuration from the specified bucket.\n        '
        try:
            self.bucket.LifecycleConfiguration().delete()
            logger.info("Deleted lifecycle configuration for bucket '%s'.", self.bucket.name)
        except ClientError:
            logger.exception("Couldn't delete lifecycle configuration for bucket '%s'.", self.bucket.name)
            raise

    def generate_presigned_post(self, object_key, expires_in):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate a presigned Amazon S3 POST request to upload a file.\n        A presigned POST can be used for a limited time to let someone without an AWS\n        account upload a file to a bucket.\n\n        :param object_key: The object key to identify the uploaded object.\n        :param expires_in: The number of seconds the presigned POST is valid.\n        :return: A dictionary that contains the URL and form fields that contain\n                 required access data.\n        '
        try:
            response = self.bucket.meta.client.generate_presigned_post(Bucket=self.bucket.name, Key=object_key, ExpiresIn=expires_in)
            logger.info('Got presigned POST URL: %s', response['url'])
        except ClientError:
            logger.exception("Couldn't get a presigned POST URL for bucket '%s' and object '%s'", self.bucket.name, object_key)
            raise
        return response

def usage_demo():
    if False:
        for i in range(10):
            print('nop')
    print('-' * 88)
    print('Welcome to the Amazon S3 bucket demo!')
    print('-' * 88)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    s3_resource = boto3.resource('s3')
    prefix = 'doc-example-bucket-'
    created_buckets = [BucketWrapper(s3_resource.Bucket(prefix + str(uuid.uuid1()))) for _ in range(3)]
    for bucket in created_buckets:
        bucket.create()
        print(f'Created bucket {bucket.name}.')
    bucket_to_delete = created_buckets.pop()
    if bucket_to_delete.exists():
        print(f'Bucket exists: {bucket_to_delete.name}.')
    bucket_to_delete.delete()
    print(f'Deleted bucket {bucket_to_delete.name}.')
    if not bucket_to_delete.exists():
        print(f'Bucket no longer exists: {bucket_to_delete.name}.')
    buckets = [b for b in BucketWrapper.list(s3_resource) if b.name.startswith(prefix)]
    for bucket in buckets:
        print(f'Got bucket {bucket.name}.')
    bucket = created_buckets[0]
    bucket.grant_log_delivery_access()
    acl = bucket.get_acl()
    print(f'Bucket {bucket.name} has ACL grants: {acl.grants}.')
    put_rules = [{'AllowedOrigins': ['http://www.example.com'], 'AllowedMethods': ['PUT', 'POST', 'DELETE'], 'AllowedHeaders': ['*']}]
    bucket.put_cors(put_rules)
    get_rules = bucket.get_cors()
    print(f'Bucket {bucket.name} has CORS rules: {json.dumps(get_rules.cors_rules)}.')
    bucket.delete_cors()
    put_policy_desc = {'Version': '2012-10-17', 'Id': str(uuid.uuid1()), 'Statement': [{'Effect': 'Allow', 'Principal': {'AWS': 'arn:aws:iam::111122223333:user/Martha'}, 'Action': ['s3:GetObject', 's3:ListBucket'], 'Resource': [f'arn:aws:s3:::{bucket.name}/*', f'arn:aws:s3:::{bucket.name}']}]}
    try:
        bucket.put_policy(put_policy_desc)
        policy = bucket.get_policy()
        print(f'Bucket {bucket.name} has policy {json.dumps(policy)}.')
        bucket.delete_policy()
    except ClientError as error:
        if error.response['Error']['Code'] == 'MalformedPolicy':
            print('*' * 88)
            print("This demo couldn't set the bucket policy because the principal user\nspecified in the demo policy does not exist. For this request to\nsucceed, you must replace the user ARN with an existing AWS user.")
            print('*' * 88)
        else:
            raise
    put_rules = [{'ID': str(uuid.uuid1()), 'Filter': {'And': {'Prefix': 'monsters/', 'Tags': [{'Key': 'type', 'Value': 'zombie'}]}}, 'Status': 'Enabled', 'Expiration': {'Days': 28}}]
    bucket.put_lifecycle_configuration(put_rules)
    get_rules = bucket.get_lifecycle_configuration()
    print(f'Bucket {bucket.name} has lifecycle configuration {json.dumps(get_rules)}.')
    bucket.delete_lifecycle_configuration()
    for bucket in created_buckets:
        bucket.delete()
        print(f'Deleted bucket {bucket.name}.')
    print('Thanks for watching!')
    print('-' * 88)
if __name__ == '__main__':
    usage_demo()