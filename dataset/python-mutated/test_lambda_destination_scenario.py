import json
import aws_cdk as cdk
import aws_cdk.aws_lambda as awslambda
import aws_cdk.aws_sns as sns
import pytest
from localstack.aws.api.lambda_ import InvocationType
from localstack.testing.aws.util import is_aws_cloud
from localstack.testing.pytest import markers
from localstack.utils.strings import short_uid, to_bytes
from localstack.utils.sync import wait_until
MAIN_FN_CODE = '\ndef handler(event, context):\n    should_fail = event.get("should_fail", "0") == "1"\n    message = event.get("message", "no message received")\n\n    if should_fail:\n        raise Exception("Failing per design.")\n\n    return {"lstest_message": message}\n'
COLLECT_FN_CODE = '\nimport json\n\ndef handler(event, context):\n    print(json.dumps(event))\n    return {"hello": "world"}  # the return value here doesn\'t really matter\n'

class TestLambdaDestinationScenario:

    @pytest.fixture(scope='class', autouse=True)
    def infrastructure(self, aws_client, infrastructure_setup):
        if False:
            for i in range(10):
                print('nop')
        infra = infrastructure_setup(namespace='LambdaDestinationScenario')
        stack = cdk.Stack(infra.cdk_app, 'LambdaTestStack')
        collect_fn = awslambda.Function(stack, 'CollectFn', code=awslambda.InlineCode(COLLECT_FN_CODE), handler='index.handler', runtime=awslambda.Runtime.PYTHON_3_10)
        topic = sns.Topic(stack, 'DestinationTopic')
        fn = awslambda.Function(stack, 'DestinationFn', code=awslambda.InlineCode(MAIN_FN_CODE), handler='index.handler', runtime=awslambda.Runtime.PYTHON_3_10)
        awslambda.EventInvokeConfig(stack, 'TopicEic', function=fn, on_success=cdk.aws_lambda_destinations.SnsDestination(topic=topic), on_failure=cdk.aws_lambda_destinations.SnsDestination(topic=topic), retry_attempts=0, max_event_age=cdk.Duration.minutes(1))
        topic.grant_publish(fn)
        collect_fn.grant_invoke(cdk.aws_iam.ServicePrincipal('sns.amazonaws.com'))
        collect_fn.add_event_source(cdk.aws_lambda_event_sources.SnsEventSource(topic))
        cdk.CfnOutput(stack, 'CollectFunctionName', value=collect_fn.function_name)
        cdk.CfnOutput(stack, 'DestinationTopicName', value=topic.topic_name)
        cdk.CfnOutput(stack, 'DestinationTopicArn', value=topic.topic_arn)
        cdk.CfnOutput(stack, 'DestinationFunctionName', value=fn.function_name)
        with infra.provisioner(skip_teardown=False) as prov:
            yield prov

    @markers.aws.validated
    @markers.snapshot.skip_snapshot_verify(paths=['$..Tags', '$..Attributes.DeliveryPolicy', '$..Attributes.EffectiveDeliveryPolicy.defaultHealthyRetryPolicy', '$..Attributes.EffectiveDeliveryPolicy.guaranteed', '$..Attributes.EffectiveDeliveryPolicy.http', '$..Attributes.EffectiveDeliveryPolicy.sicklyRetryPolicy', '$..Attributes.EffectiveDeliveryPolicy.throttlePolicy', '$..Attributes.Policy.Statement..Action', '$..Attributes.SubscriptionsConfirmed'])
    def test_infra(self, infrastructure, aws_client, snapshot):
        if False:
            while True:
                i = 10
        outputs = infrastructure.get_stack_outputs('LambdaTestStack')
        collect_fn_name = outputs['CollectFunctionName']
        main_fn_name = outputs['DestinationFunctionName']
        topic_arn = outputs['DestinationTopicArn']
        snapshot.add_transformer(snapshot.transform.lambda_api())
        snapshot.add_transformer(snapshot.transform.key_value('CodeSha256', '<code-sha-256>', reference_replacement=False))
        snapshot.add_transformer(snapshot.transform.key_value('aws:cloudformation:logical-id', 'replaced-value', reference_replacement=False), priority=-1)
        snapshot.add_transformer(snapshot.transform.key_value('aws:cloudformation:stack-id', 'replaced-value', reference_replacement=False), priority=-1)
        fn_1 = aws_client.lambda_.get_function(FunctionName=main_fn_name)
        fn_2 = aws_client.lambda_.get_function(FunctionName=collect_fn_name)
        snapshot.match('get_fn_1', fn_1)
        snapshot.match('get_fn_2', fn_2)
        eic = aws_client.lambda_.get_function_event_invoke_config(FunctionName=main_fn_name)
        assert eic['MaximumRetryAttempts'] == 0
        assert eic['DestinationConfig']['OnSuccess']['Destination'] == topic_arn
        assert eic['DestinationConfig']['OnFailure']['Destination'] == topic_arn
        snapshot.match('event_invoke_config', eic)
        topic_attr = aws_client.sns.get_topic_attributes(TopicArn=topic_arn)
        snapshot.match('topic_attributes', topic_attr)

    @markers.aws.validated
    def test_destination_sns(self, infrastructure, aws_client, snapshot):
        if False:
            for i in range(10):
                print('nop')
        outputs = infrastructure.get_stack_outputs('LambdaTestStack')
        invoke_fn_name = outputs['DestinationFunctionName']
        collect_fn_name = outputs['CollectFunctionName']
        topic_arn = outputs['DestinationTopicArn']
        msg = f'message-{short_uid()}'
        if is_aws_cloud():

            def _wait_atts():
                if False:
                    while True:
                        i = 10
                return aws_client.sns.get_topic_attributes(TopicArn=topic_arn)['Attributes']['SubscriptionsConfirmed'] == '1'
            assert wait_until(_wait_atts, strategy='static', wait=5, max_retries=60)
        response = aws_client.lambda_.invoke(FunctionName=invoke_fn_name, Payload=to_bytes(json.dumps({'message': msg, 'should_fail': '0'})), InvocationType=InvocationType.Event)
        snapshot.match('successful_invoke', response)
        response = aws_client.lambda_.invoke(FunctionName=invoke_fn_name, Payload=to_bytes(json.dumps({'message': msg, 'should_fail': '1'})), InvocationType=InvocationType.Event)
        snapshot.match('unsuccessful_invoke', response)

        def wait_for_logs():
            if False:
                for i in range(10):
                    print('nop')
            events = aws_client.logs.get_paginator('filter_log_events').paginate(logGroupName=f'/aws/lambda/{collect_fn_name}').build_full_result()['events']
            message_events = [e['message'] for e in events if msg in e['message']]
            return len(message_events) >= 2
        assert wait_until(wait_for_logs, strategy='static', max_retries=10, wait=5)