from datetime import datetime
from typing import Any, Dict, Tuple
from urllib.parse import urlsplit
import pytest
from botocore.awsrequest import AWSRequest, create_request_object
from botocore.config import Config
from botocore.model import OperationModel, ServiceModel, Shape, StructureShape
from localstack.aws.protocol.service_router import determine_aws_service_name, get_service_catalog
from localstack.http import Request
from localstack.utils.run import to_str

def _collect_operations() -> Tuple[ServiceModel, OperationModel]:
    if False:
        print('Hello World!')
    '\n    Collects all service<>operation combinations to test.\n    '
    service_catalog = get_service_catalog()
    for service_name in service_catalog.service_names:
        service = service_catalog.get(service_name)
        for operation_name in service.operation_names:
            if service.service_name in ['bedrock', 'bedrock-runtime', 'chime', 'chime-sdk-identity', 'chime-sdk-media-pipelines', 'chime-sdk-meetings', 'chime-sdk-messaging', 'chime-sdk-voice', 'codecatalyst', 'connect', 'connect-contact-lens', 'greengrassv2', 'iot1click', 'iot1click-devices', 'iot1click-projects', 'ivs', 'ivs-realtime', 'kinesis-video-archived', 'kinesis-video-archived-media', 'kinesis-video-media', 'kinesis-video-signaling', 'kinesis-video-webrtc-storage', 'kinesisvideo', 'lex-models', 'lex-runtime', 'lexv2-models', 'lexv2-runtime', 'personalize', 'personalize-events', 'personalize-runtime', 'pinpoint-sms-voice', 'sagemaker-edge', 'sagemaker-featurestore-runtime', 'sagemaker-metrics', 'sms-voice', 'sso', 'sso-oidc', 'workdocs']:
                yield pytest.param(service, service.operation_model(operation_name), marks=pytest.mark.xfail(reason=f'{service.service_name} is currently not supported by the service router'))
            elif service.service_name in ['docdb', 'neptune'] or service.service_name in 'timestream-write' or (service.service_name == 'sesv2' and operation_name == 'PutEmailIdentityDkimSigningAttributes'):
                yield pytest.param(service, service.operation_model(operation_name), marks=pytest.mark.skip(reason=f'{service.service_name} may differ due to ambiguities in the service specs'))
            else:
                yield (service, service.operation_model(operation_name))

def _botocore_request_to_localstack_request(request_object: AWSRequest) -> Request:
    if False:
        return 10
    "Converts a botocore request (AWSRequest) to our HTTP framework's Request object based on Werkzeug."
    split_url = urlsplit(request_object.url)
    path = split_url.path
    query_string = split_url.query
    body = request_object.body
    headers = request_object.headers
    return Request(method=request_object.method or 'GET', path=path, query_string=to_str(query_string), headers=dict(headers), body=body, raw_path=path)
_dummy_values = {'string': 'dummy-value', 'list': [], 'integer': 0, 'long': 0, 'timestamp': datetime.now()}

def _create_dummy_request_args(operation_model: OperationModel) -> Dict:
    if False:
        for i in range(10):
            print('nop')
    'Creates a dummy request param dict for the given operation.'
    input_shape: StructureShape = operation_model.input_shape
    if not input_shape:
        return {}
    result = {}
    for required_member in input_shape.required_members:
        required_shape: Shape = input_shape.members[required_member]
        location = required_shape.serialization.get('location')
        if location in ['uri', 'querystring', 'header', 'headers']:
            result[required_member] = _dummy_values[required_shape.type_name]
    return result

def _generate_test_name(param: Any):
    if False:
        for i in range(10):
            print('nop')
    'Simple helper function to generate readable test names.'
    if isinstance(param, ServiceModel):
        return param.service_name
    elif isinstance(param, OperationModel):
        return param.name
    return param

@pytest.mark.parametrize('service, operation', _collect_operations(), ids=_generate_test_name)
def test_service_router_works_for_every_service(service: ServiceModel, operation: OperationModel, caplog, aws_client_factory):
    if False:
        print('Hello World!')
    caplog.set_level('CRITICAL', 'botocore')
    client = aws_client_factory.get_client(service.service_name, config=Config(connect_timeout=1000, read_timeout=1000, retries={'total_max_attempts': 1}, parameter_validation=False, user_agent='aws-cli/1.33.7'))
    request_context = {'client_region': client.meta.region_name, 'client_config': client.meta.config, 'has_streaming_input': operation.has_streaming_input, 'auth_type': operation.auth_type}
    request_args = _create_dummy_request_args(operation)
    request_args = client._emit_api_params(request_args, operation, request_context)
    request_dict = client._convert_to_request_dict(request_args, operation, 'http://localhost.localstack.cloud', request_context)
    request_object = create_request_object(request_dict)
    client._request_signer.sign(operation.name, request_object)
    request: Request = _botocore_request_to_localstack_request(request_object)
    detected_service_name = determine_aws_service_name(request)
    assert detected_service_name == service.service_name

def test_endpoint_prefix_based_routing():
    if False:
        print('Hello World!')
    detected_service_name = determine_aws_service_name(Request(method='GET', path='/', headers={'Host': 'kms.localhost.localstack.cloud'}))
    assert detected_service_name == 'kms'
    detected_service_name = determine_aws_service_name(Request(method='POST', path='/app-instances', headers={'Host': 'identity-chime.localhost.localstack.cloud'}))
    assert detected_service_name == 'chime-sdk-identity'

def test_endpoint_prefix_based_routing_s3_virtual_host():
    if False:
        print('Hello World!')
    detected_service_name = determine_aws_service_name(Request(method='GET', path='/', headers={'Host': 'pictures.s3.localhost.localstack.cloud'}))
    assert detected_service_name == 's3'
    detected_service_name = determine_aws_service_name(Request(method='POST', path='/app-instances', headers={'Host': 'kms.s3.localhost.localstack.cloud'}))
    assert detected_service_name == 's3'

def test_endpoint_prefix_based_not_short_circuit_for_sqs():
    if False:
        while True:
            i = 10
    detected_service_name = determine_aws_service_name(Request(method='GET', path='/', headers={'Host': 'sqs.localhost.localstack.cloud'}))
    assert detected_service_name == 'sqs-query'
    detected_service_name = determine_aws_service_name(Request(method='GET', path='/', headers={'Host': 'sqs.localhost.localstack.cloud', 'Content-Type': 'application/x-amz-json-1.0'}))
    assert detected_service_name == 'sqs'