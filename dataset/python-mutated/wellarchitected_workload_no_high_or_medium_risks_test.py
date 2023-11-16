from re import search
from unittest import mock
from uuid import uuid4
from prowler.providers.aws.services.wellarchitected.wellarchitected_service import Workload
AWS_REGION = 'eu-west-1'
AWS_ACCOUNT_NUMBER = '123456789012'
workload_id = str(uuid4())

class Test_wellarchitected_workload_no_high_or_medium_risks:

    def test_no_wellarchitected(self):
        if False:
            i = 10
            return i + 15
        wellarchitected_client = mock.MagicMock
        wellarchitected_client.workloads = []
        with mock.patch('prowler.providers.aws.services.wellarchitected.wellarchitected_service.WellArchitected', wellarchitected_client), mock.patch('prowler.providers.aws.services.wellarchitected.wellarchitected_client.wellarchitected_client', wellarchitected_client):
            from prowler.providers.aws.services.wellarchitected.wellarchitected_workload_no_high_or_medium_risks.wellarchitected_workload_no_high_or_medium_risks import wellarchitected_workload_no_high_or_medium_risks
            check = wellarchitected_workload_no_high_or_medium_risks()
            result = check.execute()
            assert len(result) == 0

    def test_wellarchitected_no_risks(self):
        if False:
            i = 10
            return i + 15
        wellarchitected_client = mock.MagicMock
        wellarchitected_client.workloads = []
        wellarchitected_client.workloads.append(Workload(id=workload_id, arn=f'arn:aws:wellarchitected:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:workload/{workload_id}', name='test', lenses=['wellarchitected', 'serverless', 'softwareasaservice'], improvement_status='NOT_APPLICABLE', risks={}, region=AWS_REGION))
        with mock.patch('prowler.providers.aws.services.wellarchitected.wellarchitected_service.WellArchitected', wellarchitected_client), mock.patch('prowler.providers.aws.services.wellarchitected.wellarchitected_client.wellarchitected_client', wellarchitected_client):
            from prowler.providers.aws.services.wellarchitected.wellarchitected_workload_no_high_or_medium_risks.wellarchitected_workload_no_high_or_medium_risks import wellarchitected_workload_no_high_or_medium_risks
            check = wellarchitected_workload_no_high_or_medium_risks()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert search('does not contain high or medium risks', result[0].status_extended)
            assert result[0].resource_id == workload_id
            assert result[0].resource_arn == f'arn:aws:wellarchitected:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:workload/{workload_id}'

    def test_wellarchitected_no_high_medium_risks(self):
        if False:
            for i in range(10):
                print('nop')
        wellarchitected_client = mock.MagicMock
        wellarchitected_client.workloads = []
        wellarchitected_client.workloads.append(Workload(id=workload_id, arn=f'arn:aws:wellarchitected:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:workload/{workload_id}', name='test', lenses=['wellarchitected', 'serverless', 'softwareasaservice'], improvement_status='NOT_APPLICABLE', risks={'UNANSWERED': 56, 'NOT_APPLICABLE': 4}, region=AWS_REGION))
        with mock.patch('prowler.providers.aws.services.wellarchitected.wellarchitected_service.WellArchitected', wellarchitected_client), mock.patch('prowler.providers.aws.services.wellarchitected.wellarchitected_client.wellarchitected_client', wellarchitected_client):
            from prowler.providers.aws.services.wellarchitected.wellarchitected_workload_no_high_or_medium_risks.wellarchitected_workload_no_high_or_medium_risks import wellarchitected_workload_no_high_or_medium_risks
            check = wellarchitected_workload_no_high_or_medium_risks()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert search('does not contain high or medium risks', result[0].status_extended)
            assert result[0].resource_id == workload_id
            assert result[0].resource_arn == f'arn:aws:wellarchitected:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:workload/{workload_id}'

    def test_wellarchitected_with_high_medium_risks(self):
        if False:
            while True:
                i = 10
        wellarchitected_client = mock.MagicMock
        wellarchitected_client.workloads = []
        wellarchitected_client.workloads.append(Workload(id=workload_id, arn=f'arn:aws:wellarchitected:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:workload/{workload_id}', name='test', lenses=['wellarchitected', 'serverless', 'softwareasaservice'], improvement_status='NOT_APPLICABLE', risks={'UNANSWERED': 56, 'NOT_APPLICABLE': 4, 'HIGH': 10, 'MEDIUM': 20}, region=AWS_REGION))
        with mock.patch('prowler.providers.aws.services.wellarchitected.wellarchitected_service.WellArchitected', wellarchitected_client), mock.patch('prowler.providers.aws.services.wellarchitected.wellarchitected_client.wellarchitected_client', wellarchitected_client):
            from prowler.providers.aws.services.wellarchitected.wellarchitected_workload_no_high_or_medium_risks.wellarchitected_workload_no_high_or_medium_risks import wellarchitected_workload_no_high_or_medium_risks
            check = wellarchitected_workload_no_high_or_medium_risks()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert search('contains 10 high and 20 medium risks', result[0].status_extended)
            assert result[0].resource_id == workload_id
            assert result[0].resource_arn == f'arn:aws:wellarchitected:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:workload/{workload_id}'