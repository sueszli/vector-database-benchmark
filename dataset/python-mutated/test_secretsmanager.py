import json
import re
from localstack.testing.pytest import markers
from localstack.utils.strings import short_uid
from localstack.utils.sync import wait_until
SECRET_NAME = '/dev/db/pass'
TEMPLATE_GENERATE_SECRET = '\nAWSTemplateFormatVersion: \'2010-09-09\'\nResources:\n  Secret:\n    Type: \'AWS::SecretsManager::Secret\'\n    Properties:\n      Description: Aurora Password\n      Name: %s\n      GenerateSecretString:\n        SecretStringTemplate: \'{"username": "localstack-user"}\'\n        GenerateStringKey: "password"\n        PasswordLength: 30\n        IncludeSpace: false\n        ExcludePunctuation: true\nOutputs:\n  SecretARN:\n    Value: !Ref Secret\n' % SECRET_NAME
TEST_TEMPLATE_11 = '\nAWSTemplateFormatVersion: 2010-09-09\nParameters:\n  SecretName:\n    Type: String\nResources:\n  MySecret:\n    Type: AWS::SecretsManager::Secret\n    Properties:\n      Name: !Ref "SecretName"\n      Tags:\n        - Key: AppName\n          Value: AppA\n'
TEST_TEMPLATE_SECRET_POLICY = '\nResources:\n  MySecret:\n    Type: AWS::SecretsManager::Secret\n    Properties:\n          Description: This is a secret that I want to attach a resource-based policy to\n  MySecretResourcePolicy:\n    Type: AWS::SecretsManager::ResourcePolicy\n    Properties:\n      BlockPublicPolicy: True\n      SecretId:\n        Ref: MySecret\n      ResourcePolicy:\n        Version: \'2012-10-17\'\n        Statement:\n        - Resource: "*"\n          Action: secretsmanager:ReplicateSecretToRegions\n          Effect: Allow\n          Principal:\n            AWS:\n              Fn::Sub: arn:aws:iam::${AWS::AccountId}:root\nOutputs:\n  SecretId:\n    Value: !GetAtt MySecret.Id\n\n  SecretPolicyArn:\n    Value: !Ref MySecretResourcePolicy\n'

@markers.aws.unknown
def test_cfn_secretsmanager_gen_secret(deploy_cfn_template, aws_client):
    if False:
        return 10
    stack = deploy_cfn_template(template=TEMPLATE_GENERATE_SECRET)
    secret = aws_client.secretsmanager.describe_secret(SecretId='/dev/db/pass')
    assert '/dev/db/pass' == secret['Name']
    assert 'secret:/dev/db/pass' in secret['ARN']
    secret_value = aws_client.secretsmanager.get_secret_value(SecretId='/dev/db/pass')['SecretString']
    secret_json = json.loads(secret_value)
    assert 'password' in secret_json
    assert len(secret_json['password']) == 30
    assert len(stack.outputs) == 1
    output_secret_arn = stack.outputs['SecretARN']
    assert output_secret_arn == secret['ARN']
    assert re.match('.*%s-[a-zA-Z0-9]+' % SECRET_NAME, output_secret_arn)

@markers.aws.unknown
def test_cfn_handle_secretsmanager_secret(deploy_cfn_template, aws_client):
    if False:
        i = 10
        return i + 15
    secret_name = f'secret-{short_uid()}'
    stack = deploy_cfn_template(template=TEST_TEMPLATE_11, parameters={'SecretName': secret_name})
    rs = aws_client.secretsmanager.describe_secret(SecretId=secret_name)
    assert rs['Name'] == secret_name
    assert 'DeletedDate' not in rs
    aws_client.cloudformation.delete_stack(StackName=stack.stack_name)
    assert wait_until(lambda : aws_client.cloudformation.describe_stacks(StackName=stack.stack_id)['Stacks'][0]['StackStatus'] == 'DELETE_COMPLETE')
    rs = aws_client.secretsmanager.describe_secret(SecretId=secret_name)
    assert 'DeletedDate' in rs

@markers.aws.validated
def test_cfn_secret_policy(deploy_cfn_template, aws_client, snapshot):
    if False:
        print('Hello World!')
    stack = deploy_cfn_template(template=TEST_TEMPLATE_SECRET_POLICY)
    secret_id = stack.outputs['SecretId']
    snapshot.match('outputs', stack.outputs)
    secret_name = stack.outputs['SecretId'].split(':')[-1]
    snapshot.add_transformer(snapshot.transform.regex(secret_name, '<secret-name>'))
    aws_client.secretsmanager.get_resource_policy(SecretId=secret_id)