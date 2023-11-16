from localstack.aws.connect import connect_to
from localstack.services.cloudformation.deployment_utils import select_parameters
from localstack.services.cloudformation.provider_utils import generate_default_name
from localstack.services.cloudformation.service_models import GenericBaseModel
from localstack.utils.sync import poll_condition

class FirehoseDeliveryStream(GenericBaseModel):

    @staticmethod
    def cloudformation_type():
        if False:
            print('Hello World!')
        return 'AWS::KinesisFirehose::DeliveryStream'

    def fetch_state(self, stack_name, resources):
        if False:
            print('Hello World!')
        stream_name = self.props.get('DeliveryStreamName') or self.logical_resource_id
        return connect_to(aws_access_key_id=self.account_id, region_name=self.region_name).firehose.describe_delivery_stream(DeliveryStreamName=stream_name)

    @staticmethod
    def add_defaults(resource, stack_name: str):
        if False:
            print('Hello World!')
        name = resource.get('Properties', {}).get('DeliveryStreamName')
        if not name:
            resource['Properties']['DeliveryStreamName'] = generate_default_name(stack_name, resource['LogicalResourceId'])

    @staticmethod
    def get_deploy_templates():
        if False:
            i = 10
            return i + 15

        def _handle_result(account_id: str, region_name: str, result: dict, logical_resource_id: str, resource: dict):
            if False:
                for i in range(10):
                    print('nop')
            stream_name = resource['Properties']['DeliveryStreamName']
            client = connect_to(aws_access_key_id=account_id, region_name=region_name).firehose

            def check_stream_state():
                if False:
                    while True:
                        i = 10
                stream = client.describe_delivery_stream(DeliveryStreamName=stream_name)
                return stream['DeliveryStreamDescription']['DeliveryStreamStatus'] == 'ACTIVE'
            poll_condition(check_stream_state, 45, 1)
            resource['Properties']['Arn'] = result['DeliveryStreamARN']
            resource['PhysicalResourceId'] = stream_name
        return {'create': {'function': 'create_delivery_stream', 'parameters': select_parameters('DeliveryStreamName', 'DeliveryStreamType', 'S3DestinationConfiguration', 'ElasticsearchDestinationConfiguration', 'AmazonopensearchserviceDestinationConfiguration', 'DeliveryStreamEncryptionConfigurationInput', 'ExtendedS3DestinationConfiguration', 'HttpEndpointDestinationConfiguration', 'KinesisStreamSourceConfiguration', 'RedshiftDestinationConfiguration', 'SplunkDestinationConfiguration', 'Tags'), 'result_handler': _handle_result}, 'delete': {'function': 'delete_delivery_stream', 'parameters': {'DeliveryStreamName': 'DeliveryStreamName'}}}