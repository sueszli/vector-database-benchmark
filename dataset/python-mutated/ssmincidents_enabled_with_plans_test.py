from unittest import mock
from prowler.providers.aws.services.ssmincidents.ssmincidents_service import ReplicationSet, ResponsePlan
AWS_REGION = 'us-east-1'
REPLICATION_SET_ARN = 'arn:aws:ssm-incidents::111122223333:replication-set/40bd98f0-4110-2dee-b35e-b87006f9e172'
RESPONSE_PLAN_ARN = 'arn:aws:ssm-incidents::111122223333:response-plan/example-response'
AWS_ACCOUNT_NUMBER = '123456789012'

class Test_ssmincidents_enabled_with_plans:

    def test_ssmincidents_no_replicationset(self):
        if False:
            i = 10
            return i + 15
        ssmincidents_client = mock.MagicMock
        ssmincidents_client.audited_account = AWS_ACCOUNT_NUMBER
        ssmincidents_client.audited_account_arn = f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root'
        ssmincidents_client.region = AWS_REGION
        ssmincidents_client.replication_set = []
        with mock.patch('prowler.providers.aws.services.ssmincidents.ssmincidents_service.SSMIncidents', new=ssmincidents_client):
            from prowler.providers.aws.services.ssmincidents.ssmincidents_enabled_with_plans.ssmincidents_enabled_with_plans import ssmincidents_enabled_with_plans
            check = ssmincidents_enabled_with_plans()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert result[0].status_extended == 'No SSM Incidents replication set exists.'
            assert result[0].resource_id == AWS_ACCOUNT_NUMBER
            assert result[0].resource_arn == f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root'
            assert result[0].region == AWS_REGION

    def test_ssmincidents_replicationset_not_active(self):
        if False:
            for i in range(10):
                print('nop')
        ssmincidents_client = mock.MagicMock
        ssmincidents_client.audited_account = AWS_ACCOUNT_NUMBER
        ssmincidents_client.audited_account_arn = f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root'
        ssmincidents_client.region = AWS_REGION
        ssmincidents_client.replication_set = [ReplicationSet(arn=REPLICATION_SET_ARN, status='CREATING')]
        with mock.patch('prowler.providers.aws.services.ssmincidents.ssmincidents_service.SSMIncidents', new=ssmincidents_client):
            from prowler.providers.aws.services.ssmincidents.ssmincidents_enabled_with_plans.ssmincidents_enabled_with_plans import ssmincidents_enabled_with_plans
            check = ssmincidents_enabled_with_plans()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert result[0].status_extended == f'SSM Incidents replication set {REPLICATION_SET_ARN} exists but not ACTIVE.'
            assert result[0].resource_id == AWS_ACCOUNT_NUMBER
            assert result[0].resource_arn == REPLICATION_SET_ARN
            assert result[0].region == AWS_REGION

    def test_ssmincidents_replicationset_active_no_plans(self):
        if False:
            for i in range(10):
                print('nop')
        ssmincidents_client = mock.MagicMock
        ssmincidents_client.audited_account = AWS_ACCOUNT_NUMBER
        ssmincidents_client.audited_account_arn = f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root'
        ssmincidents_client.region = AWS_REGION
        ssmincidents_client.replication_set = [ReplicationSet(arn=REPLICATION_SET_ARN, status='ACTIVE')]
        ssmincidents_client.response_plans = []
        with mock.patch('prowler.providers.aws.services.ssmincidents.ssmincidents_service.SSMIncidents', new=ssmincidents_client):
            from prowler.providers.aws.services.ssmincidents.ssmincidents_enabled_with_plans.ssmincidents_enabled_with_plans import ssmincidents_enabled_with_plans
            check = ssmincidents_enabled_with_plans()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert result[0].status_extended == f'SSM Incidents replication set {REPLICATION_SET_ARN} is ACTIVE but no response plans exist.'
            assert result[0].resource_id == AWS_ACCOUNT_NUMBER
            assert result[0].resource_arn == REPLICATION_SET_ARN
            assert result[0].region == AWS_REGION

    def test_ssmincidents_replicationset_active_with_plans(self):
        if False:
            i = 10
            return i + 15
        ssmincidents_client = mock.MagicMock
        ssmincidents_client.audited_account = AWS_ACCOUNT_NUMBER
        ssmincidents_client.audited_account_arn = f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root'
        ssmincidents_client.region = AWS_REGION
        ssmincidents_client.replication_set = [ReplicationSet(arn=REPLICATION_SET_ARN, status='ACTIVE')]
        ssmincidents_client.response_plans = [ResponsePlan(arn=RESPONSE_PLAN_ARN, name='test', region=AWS_REGION)]
        with mock.patch('prowler.providers.aws.services.ssmincidents.ssmincidents_service.SSMIncidents', new=ssmincidents_client):
            from prowler.providers.aws.services.ssmincidents.ssmincidents_enabled_with_plans.ssmincidents_enabled_with_plans import ssmincidents_enabled_with_plans
            check = ssmincidents_enabled_with_plans()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert result[0].status_extended == f'SSM Incidents replication set {REPLICATION_SET_ARN} is ACTIVE and has response plans.'
            assert result[0].resource_id == AWS_ACCOUNT_NUMBER
            assert result[0].resource_arn == REPLICATION_SET_ARN
            assert result[0].region == AWS_REGION