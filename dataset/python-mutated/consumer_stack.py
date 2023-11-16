import os
import yaml
from aws_cdk import Aws, Duration, Size, Stack, aws_batch
from aws_cdk import aws_batch_alpha as batch_alpha
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_events as events
from aws_cdk import aws_events_targets as targets
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_lambda_event_sources as _event_sources
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sns as sns
from aws_cdk import aws_sns_subscriptions as subs
from aws_cdk import aws_sqs as sqs
from constructs import Construct
language_name = os.environ['LANGUAGE_NAME']

class ConsumerStack(Stack):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        if False:
            return 10
        super().__init__(scope, id, **kwargs)
        resource_config = self.get_yaml_config('../config/resources.yaml')
        topic_name = resource_config['topic_name']
        producer_bucket_name = resource_config['bucket_name']
        self.producer_account_id = resource_config['admin_acct']
        sns_topic = self.init_get_topic(topic_name)
        sqs_queue = sqs.Queue(self, f'BatchJobQueue-{language_name}')
        self.init_subscribe_sns(sqs_queue, sns_topic)
        (job_definition, job_queue) = self.init_batch_fargte()
        batch_function = self.init_batch_lambda(job_queue, job_definition)
        self.init_sqs_lambda_integration(batch_function, sqs_queue)
        self.init_log_function(producer_bucket_name)

    def get_yaml_config(self, filepath):
        if False:
            i = 10
            return i + 15
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
        return data

    def init_get_topic(self, topic_name):
        if False:
            i = 10
            return i + 15
        topic = sns.Topic(self, 'fanout-topic', topic_name=topic_name)
        return topic

    def init_batch_fargte(self):
        if False:
            for i in range(10):
                print('nop')
        batch_execution_role = iam.Role(self, f'BatchExecutionRole-{language_name}', assumed_by=iam.ServicePrincipal('ecs-tasks.amazonaws.com'), inline_policies={'BatchLoggingPolicy': iam.PolicyDocument(statements=[iam.PolicyStatement(effect=iam.Effect.ALLOW, actions=['logs:CreateLogGroup', 'logs:CreateLogStream', 'logs:PutLogEvents', 'logs:DescribeLogStreams'], resources=['arn:aws:logs:*:*:*'])])}, managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name('job-function/SystemAdministrator'), iam.ManagedPolicy.from_aws_managed_policy_name('service-role/AmazonECSTaskExecutionRolePolicy')])
        fargate_environment = batch_alpha.FargateComputeEnvironment(self, f'FargateEnv-{language_name}', vpc=ec2.Vpc.from_lookup(self, 'Vpc', is_default=True))
        container_image = ecs.EcrImage.from_registry(f'public.ecr.aws/b4v4v1s0/{language_name}:latest')
        job_definition = batch_alpha.EcsJobDefinition(self, f'JobDefinition-{language_name}', container=batch_alpha.EcsFargateContainerDefinition(self, f'ContainerDefinition-{language_name}', image=container_image, execution_role=batch_execution_role, assign_public_ip=True, memory=Size.gibibytes(2), cpu=1), timeout=Duration.minutes(500))
        job_queue = batch_alpha.JobQueue(self, f'JobQueue-{language_name}', priority=1)
        job_queue.add_compute_environment(fargate_environment, 1)
        return (job_definition, job_queue)

    def init_sqs_queue(self):
        if False:
            while True:
                i = 10
        sqs_queue = sqs.Queue(self, f'BatchJobQueue-{language_name}')
        return sqs_queue

    def init_subscribe_sns(self, sqs_queue, sns_topic):
        if False:
            print('Hello World!')
        sns_topic_role = iam.Role(self, f'SNSTopicRole-{language_name}', assumed_by=iam.ServicePrincipal('sns.amazonaws.com'), description='Allows the SNS topic to send messages to the SQS queue in this account', role_name=f'SNSTopicRole-{language_name}')
        sns_topic_policy = iam.PolicyStatement(effect=iam.Effect.ALLOW, actions=['sqs:SendMessage'], resources=[sqs_queue.queue_arn], conditions={'ArnEquals': {'aws:SourceArn': sns_topic.topic_arn}})
        subs.SqsSubscription(sqs_queue, raw_message_delivery=True).bind(sns_topic)
        sns_topic.add_subscription(subs.SqsSubscription(sqs_queue))
        sns_topic_role.add_to_policy(sns_topic_policy)
        statement = iam.PolicyStatement()
        statement.add_resources(sqs_queue.queue_arn)
        statement.add_actions('sqs:*')
        statement.add_arn_principal(f'arn:aws:iam::{self.producer_account_id}:root')
        statement.add_arn_principal(f'arn:aws:iam::{Aws.ACCOUNT_ID}:root')
        statement.add_condition('ArnLike', {'aws:SourceArn': sns_topic.topic_arn})
        sqs_queue.add_to_resource_policy(statement)

    def init_batch_lambda(self, job_queue, job_definition):
        if False:
            return 10
        execution_role = iam.Role(self, f'BatchLambdaExecutionRole-{language_name}', assumed_by=iam.ServicePrincipal('lambda.amazonaws.com'), description='Allows Lambda function to submit jobs to Batch', role_name=f'BatchLambdaExecutionRole-{language_name}')
        execution_role.add_to_policy(statement=iam.PolicyStatement(actions=['batch:*'], resources=['*']))
        execution_role.add_managed_policy(policy=iam.ManagedPolicy.from_aws_managed_policy_name('service-role/AWSLambdaBasicExecutionRole'))
        function = _lambda.Function(self, f'SubmitBatchJob-{language_name}', runtime=_lambda.Runtime.PYTHON_3_8, handler='submit_job.handler', role=execution_role, code=_lambda.Code.from_asset('lambda'), environment={'LANGUAGE_NAME': language_name, 'JOB_QUEUE': job_queue.job_queue_arn, 'JOB_DEFINITION': job_definition.job_definition_arn, 'JOB_NAME': f'job-{language_name}'})
        return function

    def init_sqs_lambda_integration(self, function, sqs_queue):
        if False:
            i = 10
            return i + 15
        function.add_event_source(_event_sources.SqsEventSource(sqs_queue))
        sqs_queue.grant_consume_messages(function)
        function.add_to_role_policy(statement=iam.PolicyStatement(actions=['sqs:ReceiveMessage'], resources=[sqs_queue.queue_arn]))
        function.add_to_role_policy(statement=iam.PolicyStatement(actions=['logs:CreateLogGroup', 'logs:CreateLogStream', 'logs:PutLogEvents'], resources=['*']))

    def init_log_function(self, producer_bucket_name):
        if False:
            print('Hello World!')
        bucket = s3.Bucket(self, 'LogBucket', versioned=False, block_public_access=s3.BlockPublicAccess.BLOCK_ALL)
        execution_role = iam.Role(self, f'LogsLambdaExecutionRole', assumed_by=iam.ServicePrincipal('lambda.amazonaws.com'), description='Allows Lambda function to get logs from CloudWatch', role_name=f'LogsLambdaExecutionRole')
        execution_role.add_managed_policy(policy=iam.ManagedPolicy.from_aws_managed_policy_name('service-role/AWSLambdaBasicExecutionRole'))
        execution_role.add_to_policy(statement=iam.PolicyStatement(actions=['logs:GetLogEvents', 'logs:DescribeLogStreams'], resources=[f'arn:aws:logs:us-east-1:{Aws.ACCOUNT_ID}:*']))
        execution_role.add_to_policy(statement=iam.PolicyStatement(actions=['s3:PutObject', 's3:GetObject'], resources=[f'{bucket.bucket_arn}/*']))
        execution_role.add_to_policy(statement=iam.PolicyStatement(actions=['s3:PutObject', 's3:PutObjectAcl'], resources=[f'arn:aws:s3:::{producer_bucket_name}/*']))
        lambda_function = _lambda.Function(self, 'BatchJobCompleteLambda', runtime=_lambda.Runtime.PYTHON_3_8, handler='export_logs.handler', role=execution_role, code=_lambda.Code.from_asset('lambda'), environment={'LANGUAGE_NAME': language_name, 'BUCKET_NAME': bucket.bucket_name, 'PRODUCER_BUCKET_NAME': f'{producer_bucket_name}'})
        batch_rule = events.Rule(self, 'BatchAllEventsRule', event_pattern=events.EventPattern(source=['aws.batch']))
        batch_rule.add_target(targets.LambdaFunction(lambda_function))