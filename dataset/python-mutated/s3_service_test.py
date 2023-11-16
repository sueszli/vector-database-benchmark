import json
from boto3 import client, session
from moto import mock_s3, mock_s3control
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.s3.s3_service import S3, S3Control
from prowler.providers.common.models import Audit_Metadata
AWS_ACCOUNT_NUMBER = '123456789012'
AWS_REGION = 'us-east-1'

class Test_S3_Service:

    def set_mocked_audit_info(self):
        if False:
            for i in range(10):
                print('nop')
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None, region_name=AWS_REGION), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_s3
    def test_service(self):
        if False:
            i = 10
            return i + 15
        audit_info = self.set_mocked_audit_info()
        s3 = S3(audit_info)
        assert s3.service == 's3'

    @mock_s3
    def test_client(self):
        if False:
            i = 10
            return i + 15
        audit_info = self.set_mocked_audit_info()
        s3 = S3(audit_info)
        assert s3.client.__class__.__name__ == 'S3'

    @mock_s3
    def test__get_session__(self):
        if False:
            i = 10
            return i + 15
        audit_info = self.set_mocked_audit_info()
        s3 = S3(audit_info)
        assert s3.session.__class__.__name__ == 'Session'

    @mock_s3
    def test_audited_account(self):
        if False:
            while True:
                i = 10
        audit_info = self.set_mocked_audit_info()
        s3 = S3(audit_info)
        assert s3.audited_account == AWS_ACCOUNT_NUMBER

    @mock_s3
    def test__list_buckets__(self):
        if False:
            i = 10
            return i + 15
        s3_client = client('s3')
        bucket_name = 'test-bucket'
        s3_client.create_bucket(Bucket=bucket_name)
        audit_info = self.set_mocked_audit_info()
        s3 = S3(audit_info)
        assert len(s3.buckets) == 1
        assert s3.buckets[0].name == bucket_name
        assert s3.buckets[0].arn == f'arn:{audit_info.audited_partition}:s3:::{bucket_name}'
        assert not s3.buckets[0].object_lock

    @mock_s3
    def test__get_bucket_versioning__(self):
        if False:
            print('Hello World!')
        s3_client = client('s3')
        bucket_name = 'test-bucket'
        s3_client.create_bucket(Bucket=bucket_name)
        s3_client.put_bucket_versioning(Bucket=bucket_name, VersioningConfiguration={'MFADelete': 'Disabled', 'Status': 'Enabled'})
        audit_info = self.set_mocked_audit_info()
        s3 = S3(audit_info)
        assert len(s3.buckets) == 1
        assert s3.buckets[0].name == bucket_name
        assert s3.buckets[0].arn == f'arn:{audit_info.audited_partition}:s3:::{bucket_name}'
        assert s3.buckets[0].versioning is True

    @mock_s3
    def test__get_bucket_acl__(self):
        if False:
            print('Hello World!')
        s3_client = client('s3')
        bucket_name = 'test-bucket'
        s3_client.create_bucket(Bucket=bucket_name)
        s3_client.put_bucket_acl(AccessControlPolicy={'Grants': [{'Grantee': {'DisplayName': 'test', 'ID': 'test_ID', 'Type': 'Group', 'URI': 'http://acs.amazonaws.com/groups/global/AllUsers'}, 'Permission': 'READ'}], 'Owner': {'DisplayName': 'test', 'ID': 'test_id'}}, Bucket=bucket_name)
        audit_info = self.set_mocked_audit_info()
        s3 = S3(audit_info)
        assert len(s3.buckets) == 1
        assert s3.buckets[0].name == bucket_name
        assert s3.buckets[0].arn == f'arn:{audit_info.audited_partition}:s3:::{bucket_name}'
        assert s3.buckets[0].acl_grantees[0].display_name == 'test'
        assert s3.buckets[0].acl_grantees[0].ID == 'test_ID'
        assert s3.buckets[0].acl_grantees[0].type == 'Group'
        assert s3.buckets[0].acl_grantees[0].URI == 'http://acs.amazonaws.com/groups/global/AllUsers'

    @mock_s3
    def test__get_bucket_logging__(self):
        if False:
            for i in range(10):
                print('nop')
        s3_client = client('s3')
        bucket_name = 'test-bucket'
        s3_client.create_bucket(Bucket=bucket_name)
        bucket_owner = s3_client.get_bucket_acl(Bucket=bucket_name)['Owner']
        s3_client.put_bucket_acl(Bucket=bucket_name, AccessControlPolicy={'Grants': [{'Grantee': {'URI': 'http://acs.amazonaws.com/groups/s3/LogDelivery', 'Type': 'Group'}, 'Permission': 'WRITE'}, {'Grantee': {'URI': 'http://acs.amazonaws.com/groups/s3/LogDelivery', 'Type': 'Group'}, 'Permission': 'READ_ACP'}, {'Grantee': {'Type': 'CanonicalUser', 'ID': bucket_owner['ID']}, 'Permission': 'FULL_CONTROL'}], 'Owner': bucket_owner})
        s3_client.put_bucket_logging(Bucket=bucket_name, BucketLoggingStatus={'LoggingEnabled': {'TargetBucket': bucket_name, 'TargetPrefix': '{}/'.format(bucket_name), 'TargetGrants': [{'Grantee': {'ID': 'SOMEIDSTRINGHERE9238748923734823917498237489237409123840983274', 'Type': 'CanonicalUser'}, 'Permission': 'READ'}, {'Grantee': {'ID': 'SOMEIDSTRINGHERE9238748923734823917498237489237409123840983274', 'Type': 'CanonicalUser'}, 'Permission': 'WRITE'}]}})
        audit_info = self.set_mocked_audit_info()
        s3 = S3(audit_info)
        assert len(s3.buckets) == 1
        assert s3.buckets[0].name == bucket_name
        assert s3.buckets[0].arn == f'arn:{audit_info.audited_partition}:s3:::{bucket_name}'
        assert s3.buckets[0].logging is True

    @mock_s3
    def test__get_bucket_policy__(self):
        if False:
            print('Hello World!')
        s3_client = client('s3')
        bucket_name = 'test-bucket'
        s3_client.create_bucket(Bucket=bucket_name)
        ssl_policy = '{"Version": "2012-10-17","Id": "PutObjPolicy","Statement": [{"Sid": "s3-bucket-ssl-requests-only","Effect": "Deny","Principal": "*","Action": "s3:GetObject","Resource": "arn:aws:s3:::bucket_test_us/*","Condition": {"Bool": {"aws:SecureTransport": "false"}}}]}'
        s3_client.put_bucket_policy(Bucket=bucket_name, Policy=ssl_policy)
        audit_info = self.set_mocked_audit_info()
        s3 = S3(audit_info)
        assert len(s3.buckets) == 1
        assert s3.buckets[0].name == bucket_name
        assert s3.buckets[0].arn == f'arn:{audit_info.audited_partition}:s3:::{bucket_name}'
        assert s3.buckets[0].policy == json.loads(ssl_policy)

    @mock_s3
    def test__get_bucket_encryption__(self):
        if False:
            i = 10
            return i + 15
        s3_client = client('s3')
        bucket_name = 'test-bucket'
        s3_client.create_bucket(Bucket=bucket_name)
        sse_config = {'Rules': [{'ApplyServerSideEncryptionByDefault': {'SSEAlgorithm': 'aws:kms', 'KMSMasterKeyID': '12345678'}}]}
        s3_client.put_bucket_encryption(Bucket=bucket_name, ServerSideEncryptionConfiguration=sse_config)
        audit_info = self.set_mocked_audit_info()
        s3 = S3(audit_info)
        assert len(s3.buckets) == 1
        assert s3.buckets[0].name == bucket_name
        assert s3.buckets[0].arn == f'arn:{audit_info.audited_partition}:s3:::{bucket_name}'
        assert s3.buckets[0].encryption == 'aws:kms'

    @mock_s3
    def test__get_bucket_ownership_controls__(self):
        if False:
            while True:
                i = 10
        s3_client = client('s3')
        bucket_name = 'test-bucket'
        s3_client.create_bucket(Bucket=bucket_name, ObjectOwnership='BucketOwnerEnforced')
        audit_info = self.set_mocked_audit_info()
        s3 = S3(audit_info)
        assert len(s3.buckets) == 1
        assert s3.buckets[0].name == bucket_name
        assert s3.buckets[0].arn == f'arn:{audit_info.audited_partition}:s3:::{bucket_name}'
        assert s3.buckets[0].ownership == 'BucketOwnerEnforced'

    @mock_s3
    def test__get_public_access_block__(self):
        if False:
            return 10
        s3_client = client('s3')
        bucket_name = 'test-bucket'
        s3_client.create_bucket(Bucket=bucket_name, ObjectOwnership='BucketOwnerEnforced')
        s3_client.put_public_access_block(Bucket=bucket_name, PublicAccessBlockConfiguration={'BlockPublicAcls': True, 'IgnorePublicAcls': True, 'BlockPublicPolicy': True, 'RestrictPublicBuckets': True})
        audit_info = self.set_mocked_audit_info()
        s3 = S3(audit_info)
        assert len(s3.buckets) == 1
        assert s3.buckets[0].name == bucket_name
        assert s3.buckets[0].arn == f'arn:{audit_info.audited_partition}:s3:::{bucket_name}'
        assert s3.buckets[0].public_access_block.block_public_acls
        assert s3.buckets[0].public_access_block.ignore_public_acls
        assert s3.buckets[0].public_access_block.block_public_policy
        assert s3.buckets[0].public_access_block.restrict_public_buckets

    @mock_s3
    def test__get_bucket_tagging__(self):
        if False:
            for i in range(10):
                print('nop')
        s3_client = client('s3')
        bucket_name = 'test-bucket'
        s3_client.create_bucket(Bucket=bucket_name)
        s3_client.put_bucket_tagging(Bucket=bucket_name, Tagging={'TagSet': [{'Key': 'test', 'Value': 'test'}]})
        audit_info = self.set_mocked_audit_info()
        s3 = S3(audit_info)
        assert len(s3.buckets) == 1
        assert s3.buckets[0].tags == [{'Key': 'test', 'Value': 'test'}]

    @mock_s3control
    def test__get_public_access_block__s3_control(self):
        if False:
            return 10
        s3control_client = client('s3control', region_name=AWS_REGION)
        s3control_client.put_public_access_block(AccountId=AWS_ACCOUNT_NUMBER, PublicAccessBlockConfiguration={'BlockPublicAcls': True, 'IgnorePublicAcls': True, 'BlockPublicPolicy': True, 'RestrictPublicBuckets': True})
        audit_info = self.set_mocked_audit_info()
        s3control = S3Control(audit_info)
        assert s3control.account_public_access_block.block_public_acls
        assert s3control.account_public_access_block.ignore_public_acls
        assert s3control.account_public_access_block.block_public_policy
        assert s3control.account_public_access_block.restrict_public_buckets

    @mock_s3
    def test__get_object_lock_configuration__(self):
        if False:
            while True:
                i = 10
        s3_client = client('s3')
        bucket_name = 'test-bucket'
        s3_client.create_bucket(Bucket=bucket_name, ObjectOwnership='BucketOwnerEnforced', ObjectLockEnabledForBucket=True)
        audit_info = self.set_mocked_audit_info()
        s3 = S3(audit_info)
        assert len(s3.buckets) == 1
        assert s3.buckets[0].name == bucket_name
        assert s3.buckets[0].arn == f'arn:{audit_info.audited_partition}:s3:::{bucket_name}'
        assert s3.buckets[0].object_lock