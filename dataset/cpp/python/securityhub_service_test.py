from unittest.mock import patch

import botocore
from boto3 import session

from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.securityhub.securityhub_service import SecurityHub
from prowler.providers.common.models import Audit_Metadata

# Mock Test Region
AWS_REGION = "eu-west-1"
AWS_ACCOUNT_NUMBER = "123456789012"
# Mocking Access Analyzer Calls
make_api_call = botocore.client.BaseClient._make_api_call


def mock_make_api_call(self, operation_name, kwarg):
    """
    We have to mock every AWS API call using Boto3

    As you can see the operation_name has the snake_case
    Rationale -> https://github.com/boto/botocore/blob/develop/botocore/client.py#L810:L816
    """
    if operation_name == "GetEnabledStandards":
        return {
            "StandardsSubscriptions": [
                {
                    "StandardsArn": "arn:aws:securityhub:::ruleset/cis-aws-foundations-benchmark/v/1.2.0",
                    "StandardsSubscriptionArn": "arn:aws:securityhub:us-east-1:0123456789012:subscription/cis-aws-foundations-benchmark/v/1.2.0",
                    "StandardsInput": {"string": "string"},
                    "StandardsStatus": "READY",
                },
            ]
        }
    if operation_name == "ListEnabledProductsForImport":
        return {
            "ProductSubscriptions": [
                "arn:aws:securityhub:us-east-1:0123456789012:product-subscription/prowler/prowler",
            ]
        }
    if operation_name == "DescribeHub":
        return {
            "HubArn": "arn:aws:securityhub:us-east-1:0123456789012:hub/default",
        }
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
class Test_SecurityHub_Service:
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

    # Test SecurityHub Client
    def test__get_client__(self):
        security_hub = SecurityHub(self.set_mocked_audit_info())
        assert (
            security_hub.regional_clients[AWS_REGION].__class__.__name__
            == "SecurityHub"
        )

    # Test SecurityHub Session
    def test__get_session__(self):
        security_hub = SecurityHub(self.set_mocked_audit_info())
        assert security_hub.session.__class__.__name__ == "Session"

    def test__describe_hub__(self):
        # Set partition for the service
        securityhub = SecurityHub(self.set_mocked_audit_info())
        assert len(securityhub.securityhubs) == 1
        assert (
            securityhub.securityhubs[0].arn
            == "arn:aws:securityhub:us-east-1:0123456789012:hub/default"
        )
        assert securityhub.securityhubs[0].id == "default"
        assert securityhub.securityhubs[0].standards == "cis-aws-foundations-benchmark "
        assert securityhub.securityhubs[0].integrations == "prowler "
