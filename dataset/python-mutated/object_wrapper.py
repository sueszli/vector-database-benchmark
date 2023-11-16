"""
Purpose
Show how to use AWS SDK for Python (Boto3) with Amazon Simple Storage Service
(Amazon S3) to perform basic object operations
"""
import json
import logging
import random
import uuid
import boto3
from botocore.exceptions import ClientError
logger = logging.getLogger(__name__)

class ObjectWrapper:
    """Encapsulates S3 object actions."""

    def __init__(self, s3_object):
        if False:
            print('Hello World!')
        '\n        :param s3_object: A Boto3 Object resource. This is a high-level resource in Boto3\n                          that wraps object actions in a class-like structure.\n        '
        self.object = s3_object
        self.key = self.object.key

    def put(self, data):
        if False:
            print('Hello World!')
        '\n        Upload data to the object.\n\n        :param data: The data to upload. This can either be bytes or a string. When this\n                     argument is a string, it is interpreted as a file name, which is\n                     opened in read bytes mode.\n        '
        put_data = data
        if isinstance(data, str):
            try:
                put_data = open(data, 'rb')
            except IOError:
                logger.exception("Expected file name or binary data, got '%s'.", data)
                raise
        try:
            self.object.put(Body=put_data)
            self.object.wait_until_exists()
            logger.info("Put object '%s' to bucket '%s'.", self.object.key, self.object.bucket_name)
        except ClientError:
            logger.exception("Couldn't put object '%s' to bucket '%s'.", self.object.key, self.object.bucket_name)
            raise
        finally:
            if getattr(put_data, 'close', None):
                put_data.close()

    def get(self):
        if False:
            return 10
        '\n        Gets the object.\n\n        :return: The object data in bytes.\n        '
        try:
            body = self.object.get()['Body'].read()
            logger.info("Got object '%s' from bucket '%s'.", self.object.key, self.object.bucket_name)
        except ClientError:
            logger.exception("Couldn't get object '%s' from bucket '%s'.", self.object.key, self.object.bucket_name)
            raise
        else:
            return body

    @staticmethod
    def list(bucket, prefix=None):
        if False:
            print('Hello World!')
        '\n        Lists the objects in a bucket, optionally filtered by a prefix.\n\n        :param bucket: The bucket to query. This is a Boto3 Bucket resource.\n        :param prefix: When specified, only objects that start with this prefix are listed.\n        :return: The list of objects.\n        '
        try:
            if not prefix:
                objects = list(bucket.objects.all())
            else:
                objects = list(bucket.objects.filter(Prefix=prefix))
            logger.info("Got objects %s from bucket '%s'", [o.key for o in objects], bucket.name)
        except ClientError:
            logger.exception("Couldn't get objects for bucket '%s'.", bucket.name)
            raise
        else:
            return objects

    def copy(self, dest_object):
        if False:
            while True:
                i = 10
        '\n        Copies the object to another bucket.\n\n        :param dest_object: The destination object initialized with a bucket and key.\n                            This is a Boto3 Object resource.\n        '
        try:
            dest_object.copy_from(CopySource={'Bucket': self.object.bucket_name, 'Key': self.object.key})
            dest_object.wait_until_exists()
            logger.info('Copied object from %s:%s to %s:%s.', self.object.bucket_name, self.object.key, dest_object.bucket_name, dest_object.key)
        except ClientError:
            logger.exception("Couldn't copy object from %s/%s to %s/%s.", self.object.bucket_name, self.object.key, dest_object.bucket_name, dest_object.key)
            raise

    def delete(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Deletes the object.\n        '
        try:
            self.object.delete()
            self.object.wait_until_not_exists()
            logger.info("Deleted object '%s' from bucket '%s'.", self.object.key, self.object.bucket_name)
        except ClientError:
            logger.exception("Couldn't delete object '%s' from bucket '%s'.", self.object.key, self.object.bucket_name)
            raise

    @staticmethod
    def delete_objects(bucket, object_keys):
        if False:
            print('Hello World!')
        '\n        Removes a list of objects from a bucket.\n        This operation is done as a batch in a single request.\n\n        :param bucket: The bucket that contains the objects. This is a Boto3 Bucket\n                       resource.\n        :param object_keys: The list of keys that identify the objects to remove.\n        :return: The response that contains data about which objects were deleted\n                 and any that could not be deleted.\n        '
        try:
            response = bucket.delete_objects(Delete={'Objects': [{'Key': key} for key in object_keys]})
            if 'Deleted' in response:
                logger.info("Deleted objects '%s' from bucket '%s'.", [del_obj['Key'] for del_obj in response['Deleted']], bucket.name)
            if 'Errors' in response:
                logger.warning("Could not delete objects '%s' from bucket '%s'.", [f"{del_obj['Key']}: {del_obj['Code']}" for del_obj in response['Errors']], bucket.name)
        except ClientError:
            logger.exception("Couldn't delete any objects from bucket %s.", bucket.name)
            raise
        else:
            return response

    @staticmethod
    def empty_bucket(bucket):
        if False:
            while True:
                i = 10
        '\n        Remove all objects from a bucket.\n\n        :param bucket: The bucket to empty. This is a Boto3 Bucket resource.\n        '
        try:
            bucket.objects.delete()
            logger.info("Emptied bucket '%s'.", bucket.name)
        except ClientError:
            logger.exception("Couldn't empty bucket '%s'.", bucket.name)
            raise

    def put_acl(self, email):
        if False:
            for i in range(10):
                print('nop')
        '\n        Applies an ACL to the object that grants read access to an AWS user identified\n        by email address.\n\n        :param email: The email address of the user to grant access.\n        '
        try:
            acl = self.object.Acl()
            grants = acl.grants if acl.grants else []
            grants.append({'Grantee': {'Type': 'AmazonCustomerByEmail', 'EmailAddress': email}, 'Permission': 'READ'})
            acl.put(AccessControlPolicy={'Grants': grants, 'Owner': acl.owner})
            logger.info('Granted read access to %s.', email)
        except ClientError:
            logger.exception("Couldn't add ACL to object '%s'.", self.object.key)
            raise

    def get_acl(self):
        if False:
            while True:
                i = 10
        '\n        Gets the ACL of the object.\n\n        :return: The ACL of the object.\n        '
        try:
            acl = self.object.Acl()
            logger.info('Got ACL for object %s owned by %s.', self.object.key, acl.owner['DisplayName'])
        except ClientError:
            logger.exception("Couldn't get ACL for object %s.", self.object.key)
            raise
        else:
            return acl

def usage_demo():
    if False:
        print('Hello World!')
    print('-' * 88)
    print('Welcome to the Amazon S3 object demo!')
    print('-' * 88)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(f'doc-example-bucket-{uuid.uuid4()}')
    try:
        bucket.create(CreateBucketConfiguration={'LocationConstraint': s3_resource.meta.client.meta.region_name})
    except ClientError as err:
        print(f"Couldn't create a bucket for the demo. Here's why: {err.response['Error']['Message']}.")
    object_key = 'doc-example-object'
    obj_wrapper = ObjectWrapper(bucket.Object(object_key))
    obj_wrapper.put(__file__)
    print(f'Put file object with key {object_key} in bucket {bucket.name}.')
    with open(__file__) as file:
        lines = file.readlines()
    line_wrappers = []
    for _ in range(10):
        line = random.randint(0, len(lines))
        line_wrapper = ObjectWrapper(bucket.Object(f'line-{line}'))
        line_wrapper.put(bytes(lines[line], 'utf-8'))
        line_wrappers.append(line_wrapper)
    print(f'Put 10 random lines from this script as objects.')
    listed_lines = ObjectWrapper.list(bucket, 'line-')
    print(f"Their keys are: {', '.join((l.key for l in listed_lines))}")
    line = line_wrappers.pop()
    line_body = line.get()
    print(f'Got object with key {line.key} and body {line_body}.')
    line.delete()
    print(f'Deleted object with key {line.key}.')
    copied_obj = bucket.Object(line_wrappers[0].key + '-copy')
    line_wrappers[0].copy(copied_obj)
    print(f'Made a copy of object {line_wrappers[0].key}, named {copied_obj.key}.')
    try:
        obj_wrapper.put_acl('arnav@example.net')
        acl = obj_wrapper.get_acl()
        print(f'Put ACL grants on object {obj_wrapper.key}: {json.dumps(acl.grants)}')
    except ClientError as error:
        if error.response['Error']['Code'] == 'UnresolvableGrantByEmailAddress':
            print('*' * 88)
            print("This demo couldn't apply the ACL to the object because the email\naddress specified as the grantee is for a test user who does not\nexist. For this request to succeed, you must replace the grantee\nemail with one for an existing AWS user.")
            print('*' * 88)
        else:
            raise
    ObjectWrapper.empty_bucket(bucket)
    print(f'Emptied bucket {bucket.name} in preparation for deleting it.')
    bucket.delete()
    print(f'Deleted bucket {bucket.name}.')
    print('Thanks for watching!')
    print('-' * 88)
if __name__ == '__main__':
    usage_demo()