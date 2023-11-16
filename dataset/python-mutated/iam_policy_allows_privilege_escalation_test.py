from json import dumps
from re import search
from unittest import mock
from boto3 import client, session
from moto import mock_iam
from prowler.providers.aws.lib.audit_info.models import AWS_Audit_Info
from prowler.providers.common.models import Audit_Metadata
AWS_REGION = 'us-east-1'
AWS_ACCOUNT_NUMBER = '123456789012'
privilege_escalation_policies_combination = {'OverPermissiveIAM': {'iam:*'}, 'IAMPut': {'iam:Put*'}, 'CreatePolicyVersion': {'iam:CreatePolicyVersion'}, 'SetDefaultPolicyVersion': {'iam:SetDefaultPolicyVersion'}, 'iam:PassRole': {'iam:PassRole'}, 'PassRole+EC2': {'iam:PassRole', 'ec2:RunInstances'}, 'PassRole+CreateLambda+Invoke': {'iam:PassRole', 'lambda:CreateFunction', 'lambda:InvokeFunction'}, 'PassRole+CreateLambda+ExistingDynamo': {'iam:PassRole', 'lambda:CreateFunction', 'lambda:CreateEventSourceMapping'}, 'PassRole+CreateLambda+NewDynamo': {'iam:PassRole', 'lambda:CreateFunction', 'lambda:CreateEventSourceMapping', 'dynamodb:CreateTable', 'dynamodb:PutItem'}, 'PassRole+GlueEndpoint': {'iam:PassRole', 'glue:CreateDevEndpoint', 'glue:GetDevEndpoint'}, 'PassRole+GlueEndpoints': {'iam:PassRole', 'glue:CreateDevEndpoint', 'glue:GetDevEndpoints'}, 'PassRole+CloudFormation': {'cloudformation:CreateStack', 'cloudformation:DescribeStacks'}, 'PassRole+DataPipeline': {'datapipeline:CreatePipeline', 'datapipeline:PutPipelineDefinition', 'datapipeline:ActivatePipeline'}, 'GlueUpdateDevEndpoint': {'glue:UpdateDevEndpoint'}, 'GlueUpdateDevEndpoints': {'glue:UpdateDevEndpoint'}, 'lambda:UpdateFunctionCode': {'lambda:UpdateFunctionCode'}, 'iam:CreateAccessKey': {'iam:CreateAccessKey'}, 'iam:CreateLoginProfile': {'iam:CreateLoginProfile'}, 'iam:UpdateLoginProfile': {'iam:UpdateLoginProfile'}, 'iam:AttachUserPolicy': {'iam:AttachUserPolicy'}, 'iam:AttachGroupPolicy': {'iam:AttachGroupPolicy'}, 'iam:AttachRolePolicy': {'iam:AttachRolePolicy'}, 'AssumeRole+AttachRolePolicy': {'sts:AssumeRole', 'iam:AttachRolePolicy'}, 'iam:PutGroupPolicy': {'iam:PutGroupPolicy'}, 'iam:PutRolePolicy': {'iam:PutRolePolicy'}, 'AssumeRole+PutRolePolicy': {'sts:AssumeRole', 'iam:PutRolePolicy'}, 'iam:PutUserPolicy': {'iam:PutUserPolicy'}, 'iam:AddUserToGroup': {'iam:AddUserToGroup'}, 'iam:UpdateAssumeRolePolicy': {'iam:UpdateAssumeRolePolicy'}, 'AssumeRole+UpdateAssumeRolePolicy': {'sts:AssumeRole', 'iam:UpdateAssumeRolePolicy'}}

class Test_iam_policy_allows_privilege_escalation:

    def set_mocked_audit_info(self):
        if False:
            while True:
                i = 10
        audit_info = AWS_Audit_Info(session_config=None, original_session=None, audit_session=session.Session(profile_name=None, botocore_session=None), audited_account=AWS_ACCOUNT_NUMBER, audited_account_arn=f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:root', audited_user_id=None, audited_partition='aws', audited_identity_arn=None, profile=None, profile_region=None, credentials=None, assumed_role_info=None, audited_regions=['us-east-1', 'eu-west-1'], organizations_metadata=None, audit_resources=None, mfa_enabled=False, audit_metadata=Audit_Metadata(services_scanned=0, expected_checks=[], completed_checks=0, audit_progress=0))
        return audit_info

    @mock_iam
    def test_iam_policy_not_allows_privilege_escalation(self):
        if False:
            return 10
        iam_client = client('iam', region_name=AWS_REGION)
        policy_name = 'policy1'
        policy_document = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 'sts:*', 'Resource': '*'}, {'Effect': 'Deny', 'Action': 'sts:*', 'Resource': '*'}, {'Effect': 'Deny', 'NotAction': 'sts:*', 'Resource': '*'}]}
        policy_arn = iam_client.create_policy(PolicyName=policy_name, PolicyDocument=dumps(policy_document))['Policy']['Arn']
        current_audit_info = self.set_mocked_audit_info()
        from prowler.providers.aws.services.iam.iam_service import IAM
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation import iam_policy_allows_privilege_escalation
            check = iam_policy_allows_privilege_escalation()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert result[0].status_extended == f'Custom Policy {policy_arn} does not allow privilege escalation.'
            assert result[0].resource_id == policy_name
            assert result[0].resource_arn == policy_arn
            assert result[0].region == AWS_REGION
            assert result[0].resource_tags == []

    @mock_iam
    def test_iam_policy_not_allows_privilege_escalation_glue_GetDevEndpoints(self):
        if False:
            while True:
                i = 10
        iam_client = client('iam', region_name=AWS_REGION)
        policy_name = 'policy1'
        policy_document = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 'lambda:*', 'Resource': '*'}, {'Effect': 'Deny', 'Action': 'lambda:InvokeFunction', 'Resource': '*'}, {'Effect': 'Deny', 'NotAction': 'glue:GetDevEndpoints', 'Resource': '*'}]}
        policy_arn = iam_client.create_policy(PolicyName=policy_name, PolicyDocument=dumps(policy_document))['Policy']['Arn']
        current_audit_info = self.set_mocked_audit_info()
        from prowler.providers.aws.services.iam.iam_service import IAM
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation import iam_policy_allows_privilege_escalation
            check = iam_policy_allows_privilege_escalation()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert result[0].status_extended == f'Custom Policy {policy_arn} does not allow privilege escalation.'
            assert result[0].resource_id == policy_name
            assert result[0].resource_arn == policy_arn
            assert result[0].region == AWS_REGION
            assert result[0].resource_tags == []

    @mock_iam
    def test_iam_policy_not_allows_privilege_escalation_dynamodb_PutItem(self):
        if False:
            print('Hello World!')
        iam_client = client('iam', region_name=AWS_REGION)
        policy_name = 'policy1'
        policy_document = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': ['lambda:*', 'iam:PassRole', 'dynamodb:PutItem', 'cloudformation:CreateStack', 'cloudformation:DescribeStacks', 'ec2:RunInstances'], 'Resource': '*'}, {'Effect': 'Deny', 'Action': ['lambda:InvokeFunction', 'cloudformation:CreateStack'], 'Resource': '*'}, {'Effect': 'Deny', 'NotAction': 'dynamodb:PutItem', 'Resource': '*'}]}
        policy_arn = iam_client.create_policy(PolicyName=policy_name, PolicyDocument=dumps(policy_document))['Policy']['Arn']
        current_audit_info = self.set_mocked_audit_info()
        from prowler.providers.aws.services.iam.iam_service import IAM
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation import iam_policy_allows_privilege_escalation
            check = iam_policy_allows_privilege_escalation()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'PASS'
            assert result[0].status_extended == f'Custom Policy {policy_arn} does not allow privilege escalation.'
            assert result[0].resource_id == policy_name
            assert result[0].resource_arn == policy_arn
            assert result[0].region == AWS_REGION
            assert result[0].resource_tags == []

    @mock_iam
    def test_iam_policy_allows_privilege_escalation_iam_all_and_ec2_RunInstances(self):
        if False:
            print('Hello World!')
        iam_client = client('iam', region_name=AWS_REGION)
        policy_name = 'policy1'
        policy_document = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': ['iam:*'], 'Resource': '*'}, {'Effect': 'Allow', 'Action': ['ec2:RunInstances'], 'Resource': '*'}]}
        policy_arn = iam_client.create_policy(PolicyName=policy_name, PolicyDocument=dumps(policy_document))['Policy']['Arn']
        current_audit_info = self.set_mocked_audit_info()
        from prowler.providers.aws.services.iam.iam_service import IAM
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation import iam_policy_allows_privilege_escalation
            check = iam_policy_allows_privilege_escalation()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert result[0].resource_id == policy_name
            assert result[0].resource_arn == policy_arn
            assert result[0].region == AWS_REGION
            assert result[0].resource_tags == []
            assert search(f'Custom Policy {policy_arn} allows privilege escalation using the following actions: ', result[0].status_extended)
            assert search('iam:PassRole', result[0].status_extended)
            assert search('ec2:RunInstances', result[0].status_extended)

    @mock_iam
    def test_iam_policy_allows_privilege_escalation_iam_PassRole(self):
        if False:
            for i in range(10):
                print('nop')
        iam_client = client('iam', region_name=AWS_REGION)
        policy_name = 'policy1'
        policy_document = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 'iam:PassRole', 'Resource': f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:role/ecs'}]}
        policy_arn = iam_client.create_policy(PolicyName=policy_name, PolicyDocument=dumps(policy_document))['Policy']['Arn']
        current_audit_info = self.set_mocked_audit_info()
        from prowler.providers.aws.services.iam.iam_service import IAM
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation import iam_policy_allows_privilege_escalation
            check = iam_policy_allows_privilege_escalation()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert result[0].resource_id == policy_name
            assert result[0].resource_arn == policy_arn
            assert result[0].region == AWS_REGION
            assert result[0].resource_tags == []
            assert search(f'Custom Policy {policy_arn} allows privilege escalation using the following actions: ', result[0].status_extended)
            assert search('iam:PassRole', result[0].status_extended)

    @mock_iam
    def test_iam_policy_allows_privilege_escalation_two_combinations(self):
        if False:
            while True:
                i = 10
        iam_client = client('iam', region_name=AWS_REGION)
        policy_name = 'policy1'
        policy_document = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': ['iam:PassRole'], 'Resource': '*'}, {'Effect': 'Allow', 'Action': ['ec2:RunInstances'], 'Resource': '*'}, {'Effect': 'Allow', 'Action': ['lambda:CreateFunction'], 'Resource': '*'}, {'Effect': 'Allow', 'Action': ['lambda:InvokeFunction'], 'Resource': '*'}]}
        policy_arn = iam_client.create_policy(PolicyName=policy_name, PolicyDocument=dumps(policy_document))['Policy']['Arn']
        current_audit_info = self.set_mocked_audit_info()
        from prowler.providers.aws.services.iam.iam_service import IAM
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation import iam_policy_allows_privilege_escalation
            check = iam_policy_allows_privilege_escalation()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert result[0].resource_id == policy_name
            assert result[0].resource_arn == policy_arn
            assert result[0].region == AWS_REGION
            assert result[0].resource_tags == []
            assert search(f'Custom Policy {policy_arn} allows privilege escalation using the following actions: ', result[0].status_extended)
            assert search('iam:PassRole', result[0].status_extended)
            assert search('lambda:InvokeFunction', result[0].status_extended)
            assert search('lambda:CreateFunction', result[0].status_extended)
            assert search('ec2:RunInstances', result[0].status_extended)

    @mock_iam
    def test_iam_policy_allows_privilege_escalation_iam_PassRole_and_other_actions(self):
        if False:
            print('Hello World!')
        iam_client = client('iam', region_name=AWS_REGION)
        policy_name = 'policy1'
        policy_document = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': 'iam:PassRole', 'Resource': f'arn:aws:iam::{AWS_ACCOUNT_NUMBER}:role/ecs'}, {'Action': 'account:GetAccountInformation', 'Effect': 'Allow', 'Resource': '*'}]}
        policy_arn = iam_client.create_policy(PolicyName=policy_name, PolicyDocument=dumps(policy_document))['Policy']['Arn']
        current_audit_info = self.set_mocked_audit_info()
        from prowler.providers.aws.services.iam.iam_service import IAM
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation import iam_policy_allows_privilege_escalation
            check = iam_policy_allows_privilege_escalation()
            result = check.execute()
            assert len(result) == 1
            assert result[0].status == 'FAIL'
            assert result[0].resource_id == policy_name
            assert result[0].resource_arn == policy_arn
            assert result[0].region == AWS_REGION
            assert result[0].resource_tags == []
            assert search(f'Custom Policy {policy_arn} allows privilege escalation using the following actions: ', result[0].status_extended)
            assert search('iam:PassRole', result[0].status_extended)

    @mock_iam
    def test_iam_policy_allows_privilege_escalation_policies_combination(self):
        if False:
            while True:
                i = 10
        current_audit_info = self.set_mocked_audit_info()
        iam_client = client('iam', region_name=AWS_REGION)
        policy_name = 'privileged_policy'
        for values in privilege_escalation_policies_combination.values():
            print(list(values))
            policy_document = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': list(values), 'Resource': '*'}]}
            policy_arn = iam_client.create_policy(PolicyName=policy_name, PolicyDocument=dumps(policy_document))['Policy']['Arn']
            from prowler.providers.aws.services.iam.iam_service import IAM
            with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation.iam_client', new=IAM(current_audit_info)):
                from prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation import iam_policy_allows_privilege_escalation
                check = iam_policy_allows_privilege_escalation()
                result = check.execute()
                assert len(result) == 1
                assert result[0].status == 'FAIL'
                assert result[0].resource_id == policy_name
                assert result[0].resource_arn == policy_arn
                assert result[0].region == AWS_REGION
                assert result[0].resource_tags == []
                assert search(f'Custom Policy {policy_arn} allows privilege escalation using the following actions: ', result[0].status_extended)
                for action in values:
                    assert search(action, result[0].status_extended)
                iam_client.delete_policy(PolicyArn=policy_arn)

    @mock_iam
    def test_iam_policy_allows_privilege_escalation_two_policies_one_good_one_bad(self):
        if False:
            return 10
        current_audit_info = self.set_mocked_audit_info()
        iam_client = client('iam', region_name=AWS_REGION)
        policy_name_1 = 'privileged_policy_1'
        policy_document_1 = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': ['ec2:RunInstances'], 'Resource': '*'}]}
        policy_name_2 = 'privileged_policy_2'
        policy_document_2 = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': ['iam:PassRole'], 'Resource': '*'}, {'Effect': 'Allow', 'Action': ['lambda:CreateFunction'], 'Resource': '*'}, {'Effect': 'Allow', 'Action': ['lambda:InvokeFunction'], 'Resource': '*'}]}
        policy_arn_1 = iam_client.create_policy(PolicyName=policy_name_1, PolicyDocument=dumps(policy_document_1))['Policy']['Arn']
        policy_arn_2 = iam_client.create_policy(PolicyName=policy_name_2, PolicyDocument=dumps(policy_document_2))['Policy']['Arn']
        from prowler.providers.aws.services.iam.iam_service import IAM
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation import iam_policy_allows_privilege_escalation
            check = iam_policy_allows_privilege_escalation()
            result = check.execute()
            assert len(result) == 2
            for finding in result:
                if finding.resource_id == policy_name_1:
                    assert finding.status == 'PASS'
                    assert finding.resource_id == policy_name_1
                    assert finding.resource_arn == policy_arn_1
                    assert finding.region == AWS_REGION
                    assert finding.resource_tags == []
                    assert finding.status_extended == f'Custom Policy {policy_arn_1} does not allow privilege escalation.'
                if finding.resource_id == policy_name_2:
                    assert finding.status == 'FAIL'
                    assert finding.resource_id == policy_name_2
                    assert finding.resource_arn == policy_arn_2
                    assert finding.region == AWS_REGION
                    assert finding.resource_tags == []
                    assert search(f'Custom Policy {policy_arn_2} allows privilege escalation using the following actions: ', finding.status_extended)
                    assert search('iam:PassRole', finding.status_extended)
                    assert search('lambda:InvokeFunction', finding.status_extended)
                    assert search('lambda:CreateFunction', finding.status_extended)

    @mock_iam
    def test_iam_policy_allows_privilege_escalation_two_bad_policies(self):
        if False:
            i = 10
            return i + 15
        current_audit_info = self.set_mocked_audit_info()
        iam_client = client('iam', region_name=AWS_REGION)
        policy_name_1 = 'privileged_policy_1'
        policy_document_1 = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': ['iam:PassRole'], 'Resource': '*'}, {'Effect': 'Allow', 'Action': ['ec2:RunInstances'], 'Resource': '*'}]}
        policy_name_2 = 'privileged_policy_2'
        policy_document_2 = {'Version': '2012-10-17', 'Statement': [{'Effect': 'Allow', 'Action': ['iam:PassRole'], 'Resource': '*'}, {'Effect': 'Allow', 'Action': ['lambda:CreateFunction'], 'Resource': '*'}, {'Effect': 'Allow', 'Action': ['lambda:InvokeFunction'], 'Resource': '*'}]}
        policy_arn_1 = iam_client.create_policy(PolicyName=policy_name_1, PolicyDocument=dumps(policy_document_1))['Policy']['Arn']
        policy_arn_2 = iam_client.create_policy(PolicyName=policy_name_2, PolicyDocument=dumps(policy_document_2))['Policy']['Arn']
        from prowler.providers.aws.services.iam.iam_service import IAM
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation import iam_policy_allows_privilege_escalation
            check = iam_policy_allows_privilege_escalation()
            result = check.execute()
            assert len(result) == 2
            for finding in result:
                if finding.resource_id == policy_name_1:
                    assert finding.status == 'FAIL'
                    assert finding.resource_id == policy_name_1
                    assert finding.resource_arn == policy_arn_1
                    assert finding.region == AWS_REGION
                    assert finding.resource_tags == []
                    assert search(f'Custom Policy {policy_arn_1} allows privilege escalation using the following actions: ', finding.status_extended)
                    assert search('iam:PassRole', finding.status_extended)
                    assert search('ec2:RunInstances', finding.status_extended)
                if finding.resource_id == policy_name_2:
                    assert finding.status == 'FAIL'
                    assert finding.resource_id == policy_name_2
                    assert finding.resource_arn == policy_arn_2
                    assert finding.region == AWS_REGION
                    assert finding.resource_tags == []
                    assert search(f'Custom Policy {policy_arn_2} allows privilege escalation using the following actions: ', finding.status_extended)
                    assert search('iam:PassRole', finding.status_extended)
                    assert search('lambda:InvokeFunction', finding.status_extended)
                    assert search('lambda:CreateFunction', finding.status_extended)

    @mock_iam
    def test_iam_policy_allows_privilege_escalation_over_permissive_policy(self):
        if False:
            for i in range(10):
                print('nop')
        current_audit_info = self.set_mocked_audit_info()
        iam_client = client('iam', region_name=AWS_REGION)
        policy_name_1 = 'privileged_policy_1'
        policy_document_1 = {'Version': '2012-10-17', 'Statement': [{'Sid': 'Statement01', 'Effect': 'Allow', 'Action': ['s3:*', 'ec2:*', 'ecr:*', 'iam:*', 'rds:*', 'dynamodb:*', 'route53:*', 'sns:*', 'sqs:*'], 'Resource': '*'}]}
        policy_arn_1 = iam_client.create_policy(PolicyName=policy_name_1, PolicyDocument=dumps(policy_document_1))['Policy']['Arn']
        from prowler.providers.aws.services.iam.iam_service import IAM
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation import iam_policy_allows_privilege_escalation
            check = iam_policy_allows_privilege_escalation()
            result = check.execute()
            assert len(result) == 1
            for finding in result:
                if finding.resource_id == policy_name_1:
                    assert finding.status == 'FAIL'
                    assert finding.resource_id == policy_name_1
                    assert finding.resource_arn == policy_arn_1
                    assert finding.region == AWS_REGION
                    assert finding.resource_tags == []
                    assert search(f'Custom Policy {policy_arn_1} allows privilege escalation using the following actions: ', finding.status_extended)
                    assert search('iam:PassRole', finding.status_extended)
                    assert search('ec2:RunInstances', finding.status_extended)

    @mock_iam
    def test_iam_policy_allows_privilege_escalation_administrator_policy(self):
        if False:
            for i in range(10):
                print('nop')
        current_audit_info = self.set_mocked_audit_info()
        iam_client = client('iam', region_name=AWS_REGION)
        policy_name_1 = 'privileged_policy_1'
        policy_document_1 = {'Version': '2012-10-17', 'Statement': [{'Sid': 'Statement01', 'Effect': 'Allow', 'Action': ['*'], 'Resource': '*'}]}
        policy_arn_1 = iam_client.create_policy(PolicyName=policy_name_1, PolicyDocument=dumps(policy_document_1))['Policy']['Arn']
        from prowler.providers.aws.services.iam.iam_service import IAM
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation import iam_policy_allows_privilege_escalation
            check = iam_policy_allows_privilege_escalation()
            result = check.execute()
            assert len(result) == 1
            for finding in result:
                if finding.resource_id == policy_name_1:
                    assert finding.status == 'FAIL'
                    assert finding.resource_id == policy_name_1
                    assert finding.resource_arn == policy_arn_1
                    assert finding.region == AWS_REGION
                    assert finding.resource_tags == []
                    assert search(f'Custom Policy {policy_arn_1} allows privilege escalation using the following actions:', finding.status_extended)
                    for permissions in privilege_escalation_policies_combination:
                        for permission in privilege_escalation_policies_combination[permissions]:
                            assert search(permission, finding.status_extended)

    @mock_iam
    def test_iam_policy_allows_privilege_escalation_iam_put(self):
        if False:
            for i in range(10):
                print('nop')
        current_audit_info = self.set_mocked_audit_info()
        iam_client = client('iam', region_name=AWS_REGION)
        policy_name_1 = 'privileged_policy_1'
        policy_document_1 = {'Version': '2012-10-17', 'Statement': [{'Sid': 'Statement01', 'Effect': 'Allow', 'Action': ['iam:Put*'], 'Resource': '*'}]}
        policy_arn_1 = iam_client.create_policy(PolicyName=policy_name_1, PolicyDocument=dumps(policy_document_1))['Policy']['Arn']
        from prowler.providers.aws.services.iam.iam_service import IAM
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation import iam_policy_allows_privilege_escalation
            check = iam_policy_allows_privilege_escalation()
            result = check.execute()
            assert len(result) == 1
            for finding in result:
                if finding.resource_id == policy_name_1:
                    assert finding.status == 'FAIL'
                    assert finding.resource_id == policy_name_1
                    assert finding.resource_arn == policy_arn_1
                    assert finding.region == AWS_REGION
                    assert finding.resource_tags == []
                    assert search(f'Custom Policy {policy_arn_1} allows privilege escalation using the following actions:', finding.status_extended)
                    assert search('iam:Put*', finding.status_extended)

    @mock_iam
    def test_iam_policy_allows_privilege_escalation_iam_wildcard(self):
        if False:
            return 10
        current_audit_info = self.set_mocked_audit_info()
        iam_client = client('iam', region_name=AWS_REGION)
        policy_name_1 = 'privileged_policy_1'
        policy_document_1 = {'Version': '2012-10-17', 'Statement': [{'Sid': 'Statement01', 'Effect': 'Allow', 'Action': ['iam:*'], 'Resource': '*'}]}
        policy_arn_1 = iam_client.create_policy(PolicyName=policy_name_1, PolicyDocument=dumps(policy_document_1))['Policy']['Arn']
        from prowler.providers.aws.services.iam.iam_service import IAM
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation import iam_policy_allows_privilege_escalation
            check = iam_policy_allows_privilege_escalation()
            result = check.execute()
            assert len(result) == 1
            for finding in result:
                if finding.resource_id == policy_name_1:
                    assert finding.status == 'FAIL'
                    assert finding.resource_id == policy_name_1
                    assert finding.resource_arn == policy_arn_1
                    assert finding.region == AWS_REGION
                    assert finding.resource_tags == []
                    assert search(f'Custom Policy {policy_arn_1} allows privilege escalation using the following actions:', finding.status_extended)
                    assert search('iam:*', finding.status_extended)

    @mock_iam
    def test_iam_policy_not_allows_privilege_escalation_custom_policy(self):
        if False:
            print('Hello World!')
        current_audit_info = self.set_mocked_audit_info()
        iam_client = client('iam', region_name=AWS_REGION)
        policy_name_1 = 'privileged_policy_1'
        policy_document_1 = {'Version': '2012-10-17', 'Statement': [{'Sid': '', 'Effect': 'Allow', 'Action': ['es:List*', 'es:Get*', 'es:Describe*'], 'Resource': '*'}, {'Sid': '', 'Effect': 'Allow', 'Action': 'es:*', 'Resource': f'arn:aws:es:{AWS_REGION}:{AWS_ACCOUNT_NUMBER}:domain/test/*'}]}
        policy_arn_1 = iam_client.create_policy(PolicyName=policy_name_1, PolicyDocument=dumps(policy_document_1))['Policy']['Arn']
        from prowler.providers.aws.services.iam.iam_service import IAM
        with mock.patch('prowler.providers.aws.lib.audit_info.audit_info.current_audit_info', new=current_audit_info), mock.patch('prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation.iam_client', new=IAM(current_audit_info)):
            from prowler.providers.aws.services.iam.iam_policy_allows_privilege_escalation.iam_policy_allows_privilege_escalation import iam_policy_allows_privilege_escalation
            check = iam_policy_allows_privilege_escalation()
            result = check.execute()
            assert len(result) == 1
            for finding in result:
                if finding.resource_id == policy_name_1:
                    assert finding.status == 'PASS'
                    assert finding.resource_id == policy_name_1
                    assert finding.resource_arn == policy_arn_1
                    assert finding.region == AWS_REGION
                    assert finding.resource_tags == []
                    assert finding.status_extended == f'Custom Policy {policy_arn_1} does not allow privilege escalation.'