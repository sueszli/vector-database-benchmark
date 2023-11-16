import json
from unittest.mock import patch

import botocore
from boto3 import session
from moto.core import DEFAULT_ACCOUNT_ID

from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.glacier.glacier_service import Glacier
from prowler.providers.common.models import Audit_Metadata

# Mock Test Region
AWS_REGION = "eu-west-1"
AWS_ACCOUNT_NUMBER = "123456789012"

# Mocking Access Analyzer Calls
make_api_call = botocore.client.BaseClient._make_api_call

TEST_VAULT_ARN = (
    f"arn:aws:glacier:{AWS_REGION}:{DEFAULT_ACCOUNT_ID}:vaults/examplevault"
)
vault_json_policy = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "cross-account-upload",
            "Principal": {"AWS": [f"arn:aws:iam::{DEFAULT_ACCOUNT_ID}:root"]},
            "Effect": "Allow",
            "Action": [
                "glacier:UploadArchive",
                "glacier:InitiateMultipartUpload",
                "glacier:AbortMultipartUpload",
                "glacier:CompleteMultipartUpload",
            ],
            "Resource": [TEST_VAULT_ARN],
        }
    ],
}


def mock_make_api_call(self, operation_name, kwarg):
    """We have to mock every AWS API call using Boto3"""
    if operation_name == "ListVaults":
        return {
            "VaultList": [
                {
                    "VaultARN": TEST_VAULT_ARN,
                    "VaultName": "examplevault",
                    "CreationDate": "2012-03-16T22:22:47.214Z",
                    "LastInventoryDate": "2012-03-21T22:06:51.218Z",
                    "NumberOfArchives": 2,
                    "SizeInBytes": 12334,
                },
            ],
        }

    if operation_name == "GetVaultAccessPolicy":
        return {"policy": {"Policy": json.dumps(vault_json_policy)}}

    if operation_name == "ListTagsForVault":
        return {"Tags": {"test": "test"}}

    return make_api_call(self, operation_name, kwarg)


# Mock generate_regional_clients()
def mock_generate_regional_clients(service, audit_info, _):
    regional_client = audit_info.audit_session.client(service, region_name=AWS_REGION)
    regional_client.region = AWS_REGION
    return {AWS_REGION: regional_client}


# Patch every AWS call using Boto3 and generate_regional_clients to have 1 client
@patch("botocore.client.BaseClient._make_api_call", new=mock_make_api_call)
@patch(
    "prowler.providers.aws.lib.service.service.generate_regional_clients",
    new=mock_generate_regional_clients,
)
class Test_Glacier_Service:
    def set_mocked_audit_info(self):
        audit_info = AWS_Audit_Info(
            session_config=None,
            original_session=None,
            audit_session=session.Session(
                profile_name=None,
                botocore_session=None,
            ),
            audited_account=AWS_ACCOUNT_NUMBER,
            audited_account_arn=f"arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root",
            audited_user_id=None,
            audited_partition="aws",
            audited_identity_arn=None,
            profile=None,
            profile_region=None,
            credentials=None,
            assumed_role_info=None,
            audited_regions=["us-east-1", "eu-west-1"],
            organizations_metadata=None,
            audit_resources=None,
            mfa_enabled=False,
            audit_metadata=Audit_Metadata(
                services_scanned=0,
                expected_checks=[],
                completed_checks=0,
                audit_progress=0,
            ),
        )

        return audit_info

    # Test Glacier Client
    def test__get_client__(self):
        glacier = Glacier(self.set_mocked_audit_info())
        assert glacier.regional_clients[AWS_REGION].__class__.__name__ == "Glacier"

    # Test Glacier Session
    def test__get_session__(self):
        glacier = Glacier(self.set_mocked_audit_info())
        assert glacier.session.__class__.__name__ == "Session"

    # Test Glacier Service
    def test__get_service__(self):
        glacier = Glacier(self.set_mocked_audit_info())
        assert glacier.service == "glacier"

    def test__list_vaults__(self):
        # Set partition for the service
        glacier = Glacier(self.set_mocked_audit_info())
        vault_name = "examplevault"
        assert len(glacier.vaults) == 1
        assert glacier.vaults[TEST_VAULT_ARN]
        assert glacier.vaults[TEST_VAULT_ARN].name == vault_name
        assert (
            glacier.vaults[TEST_VAULT_ARN].arn
            == f"arn:aws:glacier:{AWS_REGION}:{DEFAULT_ACCOUNT_ID}:vaults/examplevault"
        )
        assert glacier.vaults[TEST_VAULT_ARN].region == AWS_REGION
        assert glacier.vaults[TEST_VAULT_ARN].tags == [{"test": "test"}]

    def test__get_vault_access_policy__(self):
        # Set partition for the service
        glacier = Glacier(self.set_mocked_audit_info())
        vault_name = "examplevault"
        assert len(glacier.vaults) == 1
        assert glacier.vaults[TEST_VAULT_ARN]
        assert glacier.vaults[TEST_VAULT_ARN].name == vault_name
        assert (
            glacier.vaults[TEST_VAULT_ARN].arn
            == f"arn:aws:glacier:{AWS_REGION}:{DEFAULT_ACCOUNT_ID}:vaults/examplevault"
        )
        assert glacier.vaults[TEST_VAULT_ARN].region == AWS_REGION
        assert glacier.vaults[TEST_VAULT_ARN].access_policy == vault_json_policy
