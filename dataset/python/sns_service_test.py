from json import dumps
from unittest.mock import patch
from uuid import uuid4

import botocore
from boto3 import client, session
from moto import mock_sns

from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.sns.sns_service import SNS
from prowler.providers.common.models import Audit_Metadata

AWS_ACCOUNT_NUMBER = "123456789012"
AWS_REGION = "eu-west-1"

topic_name = "test-topic"
test_policy = {
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"AWS": f"{AWS_ACCOUNT_NUMBER}"},
            "Action": ["sns:Publish"],
            "Resource": f"arn:aws:sns:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:{topic_name}",
        }
    ]
}
kms_key_id = str(uuid4())

make_api_call = botocore.client.BaseClient._make_api_call


def mock_make_api_call(self, operation_name, kwarg):
    if operation_name == "GetTopicAttributes":
        return {
            "Attributes": {"Policy": dumps(test_policy), "KmsMasterKeyId": kms_key_id}
        }
    return make_api_call(self, operation_name, kwarg)


def mock_generate_regional_clients(service, audit_info, _):
    regional_client = audit_info.audit_session.client(service, region_name=AWS_REGION)
    regional_client.region = AWS_REGION
    return {AWS_REGION: regional_client}


@patch("botocore.client.BaseClient._make_api_call", new=mock_make_api_call)
@patch(
    "prowler.providers.aws.lib.service.service.generate_regional_clients",
    new=mock_generate_regional_clients,
)
class Test_SNS_Service:
    # Mocked Audit Info
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
            audited_regions=None,
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

    # Test SNS Service
    def test_service(self):
        audit_info = self.set_mocked_audit_info()
        sns = SNS(audit_info)
        assert sns.service == "sns"

    # Test SNS client
    def test_client(self):
        audit_info = self.set_mocked_audit_info()
        sns = SNS(audit_info)
        for reg_client in sns.regional_clients.values():
            assert reg_client.__class__.__name__ == "SNS"

    # Test SNS session
    def test__get_session__(self):
        audit_info = self.set_mocked_audit_info()
        sns = SNS(audit_info)
        assert sns.session.__class__.__name__ == "Session"

    @mock_sns
    # Test SNS session
    def test__list_topics__(self):
        sns_client = client("sns", region_name=AWS_REGION)
        sns_client.create_topic(
            Name=topic_name,
            Tags=[
                {"Key": "test", "Value": "test"},
            ],
        )

        audit_info = self.set_mocked_audit_info()
        sns = SNS(audit_info)

        assert len(sns.topics) == 1
        assert sns.topics[0].name == topic_name
        assert (
            sns.topics[0].arn
            == f"arn:aws:sns:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:{topic_name}"
        )
        assert sns.topics[0].region == AWS_REGION
        assert sns.topics[0].tags == [
            {"Key": "test", "Value": "test"},
        ]

    @mock_sns
    # Test SNS session
    def test__get_topic_attributes__(self):
        sns_client = client("sns", region_name=AWS_REGION)
        sns_client.create_topic(Name=topic_name)

        audit_info = self.set_mocked_audit_info()
        sns = SNS(audit_info)

        assert len(sns.topics) == 1
        assert (
            sns.topics[0].arn
            == f"arn:aws:sns:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:{topic_name}"
        )
        assert sns.topics[0].region == AWS_REGION
        assert sns.topics[0].policy
        assert sns.topics[0].kms_master_key_id == kms_key_id
