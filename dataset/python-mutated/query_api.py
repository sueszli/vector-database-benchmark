"""The SQS Query API allows using Queue URLs as endpoints for operations on that queue. See:
https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-making-api-requests.html. This is a
generic implementation that creates from Query API requests the respective AWS requests, and uses an aws_stack client
to make the request. """
import logging
from typing import Dict, Optional, Tuple
from urllib.parse import urlencode
from botocore.exceptions import ClientError
from botocore.model import OperationModel
from werkzeug.datastructures import Headers
from werkzeug.exceptions import NotFound
from localstack.aws.api import CommonServiceException
from localstack.aws.connect import connect_to
from localstack.aws.protocol.parser import OperationNotFoundParserError, create_parser
from localstack.aws.protocol.serializer import create_serializer
from localstack.aws.protocol.validate import MissingRequiredField, validate_request
from localstack.aws.spec import load_service
from localstack.constants import AWS_REGION_US_EAST_1, INTERNAL_AWS_ACCESS_KEY_ID, INTERNAL_AWS_SECRET_ACCESS_KEY
from localstack.http import Request, Response, Router, route
from localstack.http.dispatcher import Handler
from localstack.services.sqs.exceptions import MissingRequiredParameterException
from localstack.utils.aws.aws_stack import extract_access_key_id_from_auth_header
from localstack.utils.aws.request_context import extract_region_from_headers
from localstack.utils.strings import long_uid
LOG = logging.getLogger(__name__)
service = load_service('sqs-query')
parser = create_parser(service)
serializer = create_serializer(service)

@route('/<regex("[0-9]{12}"):account_id>/<regex("[a-zA-Z0-9_-]+(.fifo)?"):queue_name>', host='sqs.<regex("([a-z0-9-]+\\.)?"):region><regex(".*"):domain><regex("(:[0-9]{2,5})?"):port>', methods=['POST', 'GET'])
def standard_strategy_handler(request: Request, account_id: str, queue_name: str, region: str=None, domain: str=None, port: int=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Handler for modern-style endpoints which always have the region encoded.\n    See https://docs.aws.amazon.com/general/latest/gr/sqs-service.html\n    '
    return handle_request(request, region.rstrip('.'))

@route('/queue/<regex("[a-z0-9-]+"):region>/<regex("[0-9]{12}"):account_id>/<regex("[a-zA-Z0-9_-]+(.fifo)?"):queue_name>', methods=['POST', 'GET'])
def path_strategy_handler(request: Request, region, account_id: str, queue_name: str):
    if False:
        for i in range(10):
            print('nop')
    return handle_request(request, region)

@route('/<regex("[0-9]{12}"):account_id>/<regex("[a-zA-Z0-9_-]+(.fifo)?"):queue_name>', host='<regex("([a-z0-9-]+\\.)?"):region>queue.<regex(".*"):domain><regex("(:[0-9]{2,5})?"):port>', methods=['POST', 'GET'])
def domain_strategy_handler(request: Request, account_id: str, queue_name: str, region: str=None, domain: str=None, port: int=None):
    if False:
        while True:
            i = 10
    'Uses the endpoint host to extract the region. See:\n    https://docs.aws.amazon.com/general/latest/gr/sqs-service.html'
    if not region:
        region = AWS_REGION_US_EAST_1
    else:
        region = region.rstrip('.')
    return handle_request(request, region)

@route('/<regex("[0-9]{12}"):account_id>/<regex("[a-zA-Z0-9_-]+(.fifo)?"):queue_name>', methods=['POST', 'GET'])
def legacy_handler(request: Request, account_id: str, queue_name: str) -> Response:
    if False:
        i = 10
        return i + 15
    if 'X-Amz-Credential' in request.args:
        region = request.args['X-Amz-Credential'].split('/')[2]
    else:
        region = extract_region_from_headers(request.headers)
    LOG.debug('Region of queue URL %s is ambiguous, got region %s from request', request.url, region)
    return handle_request(request, region)

def register(router: Router[Handler]):
    if False:
        return 10
    '\n    Registers the query API handlers into the given router. There are four routes, one for each SQS_ENDPOINT_STRATEGY.\n\n    :param router: the router to add the handlers into.\n    '
    router.add(standard_strategy_handler)
    router.add(path_strategy_handler)
    router.add(domain_strategy_handler)
    router.add(legacy_handler)

class UnknownOperationException(Exception):
    pass

class InvalidAction(CommonServiceException):

    def __init__(self, action: str):
        if False:
            print('Hello World!')
        super().__init__('InvalidAction', f'The action {action} is not valid for this endpoint.', 400, sender_fault=True)

class BotoException(CommonServiceException):

    def __init__(self, boto_response):
        if False:
            print('Hello World!')
        error = boto_response['Error']
        super().__init__(code=error.get('Code', 'UnknownError'), status_code=boto_response['ResponseMetadata']['HTTPStatusCode'], message=error.get('Message', ''), sender_fault=error.get('Type', 'Sender') == 'Sender')

def handle_request(request: Request, region: str) -> Response:
    if False:
        for i in range(10):
            print('nop')
    if request.headers.get('Content-Type', '').lower() == 'application/x-amz-json-1.0':
        raise NotFound
    request_id = long_uid()
    try:
        (response, operation) = try_call_sqs(request, region)
        del response['ResponseMetadata']
        return serializer.serialize_to_response(response, operation, request.headers, request_id)
    except UnknownOperationException:
        return Response('<UnknownOperationException/>', 404)
    except CommonServiceException as e:
        op = service.operation_model(service.operation_names[0])
        return serializer.serialize_error_to_response(e, op, request.headers, request_id)
    except Exception as e:
        LOG.exception('exception')
        op = service.operation_model(service.operation_names[0])
        return serializer.serialize_error_to_response(CommonServiceException('InternalError', f'An internal error occurred: {e}', status_code=500), op, request.headers, request_id)

def try_call_sqs(request: Request, region: str) -> Tuple[Dict, OperationModel]:
    if False:
        print('Hello World!')
    action = request.values.get('Action')
    if not action:
        raise UnknownOperationException()
    if action in ['ListQueues', 'CreateQueue']:
        raise InvalidAction(action)
    params = {'QueueUrl': request.base_url}
    params.update(request.values)
    body = urlencode(params)
    try:
        headers = Headers(request.headers)
        headers['Content-Type'] = 'application/x-www-form-urlencoded; charset=utf-8'
        (operation, service_request) = parser.parse(Request('POST', '/', headers=headers, body=body))
        validate_request(operation, service_request).raise_first()
    except OperationNotFoundParserError:
        raise InvalidAction(action)
    except MissingRequiredField as e:
        raise MissingRequiredParameterException(f'The request must contain the parameter {e.required_name}.')
    account_id: Optional[str] = extract_access_key_id_from_auth_header(headers)
    client = connect_to(region_name=region, aws_access_key_id=account_id or INTERNAL_AWS_ACCESS_KEY_ID, aws_secret_access_key=INTERNAL_AWS_SECRET_ACCESS_KEY).sqs_query
    try:
        boto_response = client._make_api_call(operation.name, service_request)
    except ClientError as e:
        raise BotoException(e.response) from e
    return (boto_response, operation)