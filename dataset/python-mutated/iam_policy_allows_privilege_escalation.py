from prowler.lib.check.models import Check, Check_Report_AWS
from prowler.providers.aws.services.iam.iam_client import iam_client

class iam_policy_allows_privilege_escalation(Check):

    def execute(self) -> Check_Report_AWS:
        if False:
            print('Hello World!')
        privilege_escalation_policies_combination = {'OverPermissiveIAM': {'iam:*'}, 'IAMPut': {'iam:Put*'}, 'CreatePolicyVersion': {'iam:CreatePolicyVersion'}, 'SetDefaultPolicyVersion': {'iam:SetDefaultPolicyVersion'}, 'iam:PassRole': {'iam:PassRole'}, 'PassRole+EC2': {'iam:PassRole', 'ec2:RunInstances'}, 'PassRole+CreateLambda+Invoke': {'iam:PassRole', 'lambda:CreateFunction', 'lambda:InvokeFunction'}, 'PassRole+CreateLambda+ExistingDynamo': {'iam:PassRole', 'lambda:CreateFunction', 'lambda:CreateEventSourceMapping'}, 'PassRole+CreateLambda+NewDynamo': {'iam:PassRole', 'lambda:CreateFunction', 'lambda:CreateEventSourceMapping', 'dynamodb:CreateTable', 'dynamodb:PutItem'}, 'PassRole+GlueEndpoint': {'iam:PassRole', 'glue:CreateDevEndpoint', 'glue:GetDevEndpoint'}, 'PassRole+GlueEndpoints': {'iam:PassRole', 'glue:CreateDevEndpoint', 'glue:GetDevEndpoints'}, 'PassRole+CloudFormation': {'cloudformation:CreateStack', 'cloudformation:DescribeStacks'}, 'PassRole+DataPipeline': {'datapipeline:CreatePipeline', 'datapipeline:PutPipelineDefinition', 'datapipeline:ActivatePipeline'}, 'GlueUpdateDevEndpoint': {'glue:UpdateDevEndpoint'}, 'GlueUpdateDevEndpoints': {'glue:UpdateDevEndpoint'}, 'lambda:UpdateFunctionCode': {'lambda:UpdateFunctionCode'}, 'iam:CreateAccessKey': {'iam:CreateAccessKey'}, 'iam:CreateLoginProfile': {'iam:CreateLoginProfile'}, 'iam:UpdateLoginProfile': {'iam:UpdateLoginProfile'}, 'iam:AttachUserPolicy': {'iam:AttachUserPolicy'}, 'iam:AttachGroupPolicy': {'iam:AttachGroupPolicy'}, 'iam:AttachRolePolicy': {'iam:AttachRolePolicy'}, 'AssumeRole+AttachRolePolicy': {'sts:AssumeRole', 'iam:AttachRolePolicy'}, 'iam:PutGroupPolicy': {'iam:PutGroupPolicy'}, 'iam:PutRolePolicy': {'iam:PutRolePolicy'}, 'AssumeRole+PutRolePolicy': {'sts:AssumeRole', 'iam:PutRolePolicy'}, 'iam:PutUserPolicy': {'iam:PutUserPolicy'}, 'iam:AddUserToGroup': {'iam:AddUserToGroup'}, 'iam:UpdateAssumeRolePolicy': {'iam:UpdateAssumeRolePolicy'}, 'AssumeRole+UpdateAssumeRolePolicy': {'sts:AssumeRole', 'iam:UpdateAssumeRolePolicy'}}
        findings = []
        for policy in iam_client.policies:
            if policy.type == 'Custom':
                report = Check_Report_AWS(self.metadata())
                report.resource_id = policy.name
                report.resource_arn = policy.arn
                report.region = iam_client.region
                report.resource_tags = policy.tags
                report.status = 'PASS'
                report.status_extended = f'Custom Policy {report.resource_arn} does not allow privilege escalation.'
                allowed_actions = set()
                denied_actions = set()
                denied_not_actions = set()
                if policy.document:
                    if not isinstance(policy.document['Statement'], list):
                        policy_statements = [policy.document['Statement']]
                    else:
                        policy_statements = policy.document['Statement']
                    for statements in policy_statements:
                        if statements['Effect'] == 'Allow':
                            if 'Action' in statements:
                                if type(statements['Action']) is str:
                                    allowed_actions.add(statements['Action'])
                                if type(statements['Action']) is list:
                                    allowed_actions.update(statements['Action'])
                        if statements['Effect'] == 'Deny':
                            if 'Action' in statements:
                                if type(statements['Action']) is str:
                                    denied_actions.add(statements['Action'])
                                if type(statements['Action']) is list:
                                    denied_actions.update(statements['Action'])
                            if 'NotAction' in statements:
                                if type(statements['NotAction']) is str:
                                    denied_not_actions.add(statements['NotAction'])
                                if type(statements['NotAction']) is list:
                                    denied_not_actions.update(statements['NotAction'])
                    left_actions = allowed_actions.difference(denied_actions)
                    if denied_not_actions:
                        privileged_actions = left_actions.intersection(denied_not_actions)
                    else:
                        privileged_actions = left_actions
                    policies_combination = set()
                    for values in privilege_escalation_policies_combination.values():
                        for val in values:
                            val_set = set()
                            val_set.add(val)
                            if privileged_actions.intersection(val_set) == val_set:
                                policies_combination.add(val)
                            else:
                                for permission in privileged_actions:
                                    api_action = permission.split(':')
                                    if len(api_action) == 2:
                                        api = api_action[0]
                                        action = api_action[1]
                                        if action == '*':
                                            val_api = val.split(':')[0]
                                            if api == val_api:
                                                policies_combination.add(val)
                                    elif len(api_action) == 1:
                                        api = api_action[0]
                                        if api == '*':
                                            policies_combination.add(val)
                    combos = set()
                    for (key, values) in privilege_escalation_policies_combination.items():
                        intersection = policies_combination.intersection(values)
                        if intersection == values:
                            combos.add(key)
                    if len(combos) != 0:
                        report.status = 'FAIL'
                        policies_affected = ''
                        for key in combos:
                            policies_affected += str(privilege_escalation_policies_combination[key]) + ' '
                        report.status_extended = f'Custom Policy {report.resource_arn} allows privilege escalation using the following actions: {policies_affected}'.rstrip() + '.'
                findings.append(report)
        return findings