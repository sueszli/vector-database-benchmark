import json
import pytest
from localstack.aws.connect import ServiceLevelClientFactory
from localstack.testing.pytest import markers
from localstack.testing.pytest.snapshot import is_aws

@pytest.mark.skipif(not is_aws(), reason='Test only works on AWS')
@markers.aws.unknown
def test_cloudtrail_trace_example(cfn_store_events_role_arn, aws_client: ServiceLevelClientFactory, deploy_cfn_template):
    if False:
        return 10
    '\n    Example test to demonstrate capturing CloudFormation events using CloudTrail.\n    '
    template = json.dumps({'Resources': {'MyTopic': {'Type': 'AWS::SNS::Topic'}}, 'Outputs': {'TopicArn': {'Value': {'Fn::GetAtt': ['MyTopic', 'TopicArn']}}}})
    stack = deploy_cfn_template(template=template, role_arn=cfn_store_events_role_arn)
    aws_client.sns.get_topic_attributes(TopicArn=stack.outputs['TopicArn'])