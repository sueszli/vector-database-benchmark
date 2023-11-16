from unittest.mock import patch
import botocore
import yaml
from boto3 import client, session
from moto import mock_ssm
from moto.core import DEFAULT_ACCOUNT_ID
from prowler.providers.aws.lib.audit_info.audit_info import AWS_Audit_Info
from prowler.providers.aws.services.ssm.ssm_service import SSM, ResourceStatus
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'eu-west-1'
make_api_call = botocore.client.BaseClient._make_api_call

def mock_make_api_call(self, operation_name, kwarg):
    if False:
        return 10
    'We have to mock every AWS API call using Boto3'
    if operation_name == 'ListResourceComplianceSummaries':
        return {'ResourceComplianceSummaryItems': [{'ComplianceType': 'Association', 'ResourceType': 'ManagedInstance', 'ResourceId': 'i-1234567890abcdef0', 'Status': 'COMPLIANT', 'OverallSeverity': 'UNSPECIFIED', 'ExecutionSummary': {'ExecutionTime': 1550509273.0}, 'CompliantSummary': {'CompliantCount': 2, 'SeveritySummary': {'CriticalCount': 0, 'HighCount': 0, 'MediumCount': 0, 'LowCount': 0, 'InformationalCount': 0, 'UnspecifiedCount': 2}}, 'NonCompliantSummary': {'NonCompliantCount': 0, 'SeveritySummary': {'CriticalCount': 0, 'HighCount': 0, 'MediumCount': 0, 'LowCount': 0, 'InformationalCount': 0, 'UnspecifiedCount': 0}}}]}
    if operation_name == 'DescribeInstanceInformation':
        return {'InstanceInformationList': [{'InstanceId': 'test-instance-id'}]}
    return make_api_call(self, operation_name, kwarg)

def mock_generate_regional_clients(service, audit_info, _):
    if False:
        while True:
            i = 10
    regional_client = audit_info.audit_session.client(service, region_name=AWS_REGION)
    regional_client.region = AWS_REGION
    return {AWS_REGION: regional_client}
ssm_document_yaml = '\nschemaVersion: "2.2"\ndescription: "Sample Yaml"\nparameters:\n  Parameter1:\n    type: "Integer"\n    default: 3\n    description: "Command Duration."\n    allowedValues: [1,2,3,4]\n  Parameter2:\n    type: "String"\n    default: "def"\n    description:\n    allowedValues: ["abc", "def", "ghi"]\n    allowedPattern: r"^[a-zA-Z0-9_-.]{3,128}$"\n  Parameter3:\n    type: "Boolean"\n    default: false\n    description: "A boolean"\n    allowedValues: [True, False]\n  Parameter4:\n    type: "StringList"\n    default: ["abc", "def"]\n    description: "A string list"\n  Parameter5:\n    type: "StringMap"\n    default:\n      NotificationType: Command\n      NotificationEvents:\n      - Failed\n      NotificationArn: "$dependency.topicArn"\n    description:\n  Parameter6:\n    type: "MapList"\n    default:\n    - DeviceName: "/dev/sda1"\n      Ebs:\n        VolumeSize: \'50\'\n    - DeviceName: "/dev/sdm"\n      Ebs:\n        VolumeSize: \'100\'\n    description:\nmainSteps:\n  - action: "aws:runShellScript"\n    name: "sampleCommand"\n    inputs:\n      runCommand:\n        - "echo hi"\n'

@patch('botocore.client.BaseClient._make_api_call', new=mock_make_api_call)
@patch('prowler.providers.aws.lib.service.service.generate_regional_clients', new=mock_generate_regional_clients)
class Test_SSM_Service:

    def set_mocked_audit_info(self):
        if False:
            i = 10
            return i + 15
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=DEFAULT_ACCOUNT_ID, audited_account_arn=f'arn:aws:iam::{DEFAULT_ACCOUNT_ID}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=AWS_REGION, credentials=None, assumed_role_info=None, audited_regions=None, organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_ssm
    def test__get_client__(self):
        if False:
            return 10
        ssm = SSM(self.set_mocked_audit_info())
        assert ssm.regional_clients[AWS_REGION].__class__.__name__ == 'SSM'

    @mock_ssm
    def test__get_session__(self):
        if False:
            print('Hello World!')
        ssm = SSM(self.set_mocked_audit_info())
        assert ssm.session.__class__.__name__ == 'Session'

    @mock_ssm
    def test__get_service__(self):
        if False:
            return 10
        ssm = SSM(self.set_mocked_audit_info())
        assert ssm.service == 'ssm'

    @mock_ssm
    def test__list_documents__(self):
        if False:
            while True:
                i = 10
        ssm_client = client('ssm', region_name=AWS_REGION)
        ssm_document_name = 'test-document'
        _ = ssm_client.create_document(Content=ssm_document_yaml, Name=ssm_document_name, DocumentType='Command', DocumentFormat='YAML', Tags=[{'Key': 'test', 'Value': 'test'}])
        ssm_client.modify_document_permission(Name=ssm_document_name, PermissionType='Share', AccountIdsToAdd=[DEFAULT_ACCOUNT_ID])
        ssm = SSM(self.set_mocked_audit_info())
        document_arn = f'arn:aws:ssm:{AWS_REGION}:{DEFAULT_ACCOUNT_ID}:document/{ssm_document_name}'
        assert len(ssm.documents) == 1
        assert ssm.documents
        assert ssm.documents[document_arn]
        assert ssm.documents[document_arn].arn == document_arn
        assert ssm.documents[document_arn].name == ssm_document_name
        assert ssm.documents[document_arn].region == AWS_REGION
        assert ssm.documents[document_arn].tags == [{'Key': 'test', 'Value': 'test'}]
        assert ssm.documents[document_arn].content == yaml.safe_load(ssm_document_yaml)
        assert ssm.documents[document_arn].account_owners == [DEFAULT_ACCOUNT_ID]

    @mock_ssm
    def test__list_resource_compliance_summaries__(self):
        if False:
            i = 10
            return i + 15
        ssm = SSM(self.set_mocked_audit_info())
        instance_id = 'i-1234567890abcdef0'
        assert len(ssm.compliance_resources) == 1
        assert ssm.compliance_resources
        assert ssm.compliance_resources[instance_id]
        assert ssm.compliance_resources[instance_id].id == instance_id
        assert ssm.compliance_resources[instance_id].region == AWS_REGION
        assert ssm.compliance_resources[instance_id].status == ResourceStatus.COMPLIANT