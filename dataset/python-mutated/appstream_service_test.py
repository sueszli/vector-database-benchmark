from unittest.mock import patch
import botocore
from boto3 import session
from moto.core import DEFAULT_ACCOUNT_ID
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.aws.services.appstream.appstream_service import AppStream
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'eu-west-1'
AWS_ACCOUNT_NUMBER = '123456789012'
make_api_call = botocore.client.BaseClient._make_api_call

def mock_make_api_call(self, operation_name, kwarg):
    if False:
        return 10
    '\n    We have to mock every AWS API call using Boto3\n\n    As you can see the operation_name has the list_analyzers snake_case form but\n    we are using the ListAnalyzers form.\n    Rationale -> https://github.com/boto/botocore/blob/develop/botocore/client.py#L810:L816\n    '
    if operation_name == 'DescribeFleets':
        return {'Fleets': [{'Arn': f'arn:aws:appstream:{AWS_REGION}:{DEFAULT_ACCOUNT_ID}:fleet/test-prowler3-0', 'Name': 'test-prowler3-0', 'MaxUserDurationInSeconds': 100, 'DisconnectTimeoutInSeconds': 900, 'IdleDisconnectTimeoutInSeconds': 900, 'EnableDefaultInternetAccess': False}, {'Arn': f'arn:aws:appstream:{AWS_REGION}:{DEFAULT_ACCOUNT_ID}:fleet/test-prowler3-1', 'Name': 'test-prowler3-1', 'MaxUserDurationInSeconds': 57600, 'DisconnectTimeoutInSeconds': 900, 'IdleDisconnectTimeoutInSeconds': 900, 'EnableDefaultInternetAccess': True}]}
    if operation_name == 'ListTagsForResource':
        return {'Tags': {'test': 'test'}}
    return make_api_call(self, operation_name, kwarg)

def mock_generate_regional_clients(service, audit_info, _):
    if False:
        print('Hello World!')
    regional_client = audit_info.audit_session.client(service, region_name=AWS_REGION)
    regional_client.region = AWS_REGION
    return {AWS_REGION: regional_client}

@patch('botocore.client.BaseClient._make_api_call', new=mock_make_api_call)
@patch('prowler.providers.aws.lib.service.service.generate_regional_clients', new=mock_generate_regional_clients)
class Test_AppStream_Service:

    def set_mocked_audit_info(self):
        if False:
            return 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=['us-east-1', 'eu-west-1'], organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    def test__get_client__(self):
        if False:
            return 10
        appstream = AppStream(self.set_mocked_audit_info())
        assert appstream.regional_clients[AWS_REGION].__class__.__name__ == 'AppStream'

    def test__get_session__(self):
        if False:
            print('Hello World!')
        appstream = AppStream(self.set_mocked_audit_info())
        assert appstream.session.__class__.__name__ == 'Session'

    def test__get_service__(self):
        if False:
            for i in range(10):
                print('nop')
        appstream = AppStream(self.set_mocked_audit_info())
        assert appstream.service == 'appstream'

    def test__describe_fleets__(self):
        if False:
            for i in range(10):
                print('nop')
        appstream = AppStream(self.set_mocked_audit_info())
        assert len(appstream.fleets) == 2
        assert appstream.fleets[0].arn == f'arn:aws:appstream:{AWS_REGION}:{DEFAULT_ACCOUNT_ID}:fleet/test-prowler3-0'
        assert appstream.fleets[0].name == 'test-prowler3-0'
        assert appstream.fleets[0].max_user_duration_in_seconds == 100
        assert appstream.fleets[0].disconnect_timeout_in_seconds == 900
        assert appstream.fleets[0].idle_disconnect_timeout_in_seconds == 900
        assert appstream.fleets[0].enable_default_internet_access is False
        assert appstream.fleets[0].region == AWS_REGION
        assert appstream.fleets[1].arn == f'arn:aws:appstream:{AWS_REGION}:{DEFAULT_ACCOUNT_ID}:fleet/test-prowler3-1'
        assert appstream.fleets[1].name == 'test-prowler3-1'
        assert appstream.fleets[1].max_user_duration_in_seconds == 57600
        assert appstream.fleets[1].disconnect_timeout_in_seconds == 900
        assert appstream.fleets[1].idle_disconnect_timeout_in_seconds == 900
        assert appstream.fleets[1].enable_default_internet_access is True
        assert appstream.fleets[1].region == AWS_REGION

    def test__list_tags_for_resource__(self):
        if False:
            for i in range(10):
                print('nop')
        appstream = AppStream(self.set_mocked_audit_info())
        assert len(appstream.fleets) == 2
        assert appstream.fleets[0].tags == [{'test': 'test'}]
        assert appstream.fleets[1].tags == [{'test': 'test'}]