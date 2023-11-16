import os
import pytest
from localstack.testing.pytest import markers
from localstack.utils.files import load_file
from localstack.utils.strings import short_uid
from localstack.utils.sync import wait_until

@pytest.fixture
def wait_stack_set_operation(aws_client):
    if False:
        print('Hello World!')

    def waiter(stack_set_name: str, operation_id: str):
        if False:
            while True:
                i = 10

        def _operation_is_ready():
            if False:
                print('Hello World!')
            operation = aws_client.cloudformation.describe_stack_set_operation(StackSetName=stack_set_name, OperationId=operation_id)
            return operation['StackSetOperation']['Status'] not in ['RUNNING', 'STOPPING']
        wait_until(_operation_is_ready)
    return waiter

@markers.aws.validated
def test_create_stack_set_with_stack_instances(account_id, region, aws_client, snapshot, wait_stack_set_operation):
    if False:
        i = 10
        return i + 15
    snapshot.add_transformer(snapshot.transform.key_value('StackSetId', 'stack-set-id'))
    stack_set_name = f'StackSet-{short_uid()}'
    template_body = load_file(os.path.join(os.path.dirname(__file__), '../../../templates/s3_cors_bucket.yaml'))
    result = aws_client.cloudformation.create_stack_set(StackSetName=stack_set_name, TemplateBody=template_body)
    snapshot.match('create_stack_set', result)
    create_instances_result = aws_client.cloudformation.create_stack_instances(StackSetName=stack_set_name, Accounts=[account_id], Regions=[region])
    snapshot.match('create_stack_instances', create_instances_result)
    wait_stack_set_operation(stack_set_name, create_instances_result['OperationId'])
    create_instances_result = aws_client.cloudformation.create_stack_instances(StackSetName=stack_set_name, Accounts=[account_id], Regions=[region])
    assert 'OperationId' in create_instances_result
    wait_stack_set_operation(stack_set_name, create_instances_result['OperationId'])
    delete_instances_result = aws_client.cloudformation.delete_stack_instances(StackSetName=stack_set_name, Accounts=[account_id], Regions=[region], RetainStacks=False)
    wait_stack_set_operation(stack_set_name, delete_instances_result['OperationId'])
    aws_client.cloudformation.delete_stack_set(StackSetName=stack_set_name)