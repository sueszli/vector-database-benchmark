import json
import logging
from jsonschema import ValidationError, validate
from requests.models import Response
from werkzeug.exceptions import NotFound
from localstack.aws.connect import connect_to
from localstack.constants import APPLICATION_JSON
from localstack.services.apigateway import helpers
from localstack.services.apigateway.context import ApiInvocationContext
from localstack.services.apigateway.helpers import EMPTY_MODEL, ModelResolver, get_apigateway_store_for_invocation, get_cors_response, make_error_response
from localstack.services.apigateway.integration import ApiGatewayIntegrationError, DynamoDBIntegration, EventBridgeIntegration, HTTPIntegration, KinesisIntegration, LambdaIntegration, LambdaProxyIntegration, MockIntegration, S3Integration, SNSIntegration, SQSIntegration, StepFunctionIntegration
from localstack.services.apigateway.models import ApiGatewayStore
from localstack.utils.aws.aws_responses import requests_response
LOG = logging.getLogger(__name__)

class AuthorizationError(Exception):
    message: str
    status_code: int

    def __init__(self, message: str, status_code: int):
        if False:
            i = 10
            return i + 15
        super().__init__(message)
        self.message = message
        self.status_code = status_code

    def to_response(self):
        if False:
            for i in range(10):
                print('nop')
        return requests_response({'message': self.message}, status_code=self.status_code)

class RequestValidator:
    __slots__ = ['context', 'rest_api_container']

    def __init__(self, context: ApiInvocationContext, store: ApiGatewayStore=None):
        if False:
            print('Hello World!')
        self.context = context
        store = store or get_apigateway_store_for_invocation(context=context)
        if not (container := store.rest_apis.get(context.api_id)):
            raise NotFound()
        self.rest_api_container = container

    def is_request_valid(self) -> bool:
        if False:
            print('Hello World!')
        if self.context.resource is None or 'resourceMethods' not in self.context.resource:
            return True
        resource_methods = self.context.resource['resourceMethods']
        if self.context.method not in resource_methods and 'ANY' not in resource_methods:
            return True
        resource = resource_methods.get(self.context.method, resource_methods.get('ANY', {}))
        if not (resource.get('requestValidatorId') or '').strip():
            return True
        validator = self.rest_api_container.validators.get(resource['requestValidatorId'])
        if not validator:
            return True
        if self.should_validate_body(validator):
            is_body_valid = self.validate_body(resource)
            if not is_body_valid:
                return is_body_valid
        if self.should_validate_request(validator):
            is_valid_parameters = self.validate_parameters_and_headers(resource)
            if not is_valid_parameters:
                return is_valid_parameters
        return True

    def validate_body(self, resource):
        if False:
            for i in range(10):
                print('nop')
        if not (request_models := resource.get('requestModels')):
            model_name = EMPTY_MODEL
        else:
            model_name = request_models.get(APPLICATION_JSON, EMPTY_MODEL)
        model_resolver = ModelResolver(rest_api_container=self.rest_api_container, model_name=model_name)
        resolved_schema = model_resolver.get_resolved_model()
        if not resolved_schema:
            LOG.exception('An exception occurred while trying to validate the request: could not find the model')
            return False
        try:
            validate(instance=json.loads(self.context.data or '{}'), schema=resolved_schema)
            return True
        except ValidationError as e:
            LOG.warning('failed to validate request body %s', e)
            return False
        except json.JSONDecodeError as e:
            LOG.warning('failed to validate request body, request data is not valid JSON %s', e)
            return False

    def validate_parameters_and_headers(self, resource):
        if False:
            while True:
                i = 10
        return True

    @staticmethod
    def should_validate_body(validator):
        if False:
            for i in range(10):
                print('nop')
        return validator['validateRequestBody']

    @staticmethod
    def should_validate_request(validator):
        if False:
            while True:
                i = 10
        return validator.get('validateRequestParameters')

def validate_api_key(api_key: str, invocation_context: ApiInvocationContext):
    if False:
        return 10
    usage_plan_ids = []
    client = connect_to(aws_access_key_id=invocation_context.account_id, region_name=invocation_context.region_name).apigateway
    usage_plans = client.get_usage_plans()
    for item in usage_plans.get('items', []):
        api_stages = item.get('apiStages', [])
        usage_plan_ids.extend((item.get('id') for api_stage in api_stages if api_stage.get('stage') == invocation_context.stage and api_stage.get('apiId') == invocation_context.api_id))
    for usage_plan_id in usage_plan_ids:
        usage_plan_keys = client.get_usage_plan_keys(usagePlanId=usage_plan_id)
        for key in usage_plan_keys.get('items', []):
            if key.get('value') == api_key:
                api_key = client.get_api_key(apiKey=key.get('id'))
                return api_key.get('enabled') in ('true', True)
    return False

def is_api_key_valid(invocation_context: ApiInvocationContext) -> bool:
    if False:
        print('Hello World!')
    client = connect_to(aws_access_key_id=invocation_context.account_id, region_name=invocation_context.region_name).apigateway
    rest_api = client.get_rest_api(restApiId=invocation_context.api_id)
    api_key_source = rest_api.get('apiKeySource')
    match api_key_source:
        case 'HEADER':
            api_key = invocation_context.headers.get('X-API-Key')
            return validate_api_key(api_key, invocation_context) if api_key else False
        case 'AUTHORIZER':
            api_key = invocation_context.auth_identity.get('apiKey')
            return validate_api_key(api_key, invocation_context) if api_key else False

def update_content_length(response: Response):
    if False:
        while True:
            i = 10
    if response and response.content is not None:
        response.headers['Content-Length'] = str(len(response.content))

def invoke_rest_api_from_request(invocation_context: ApiInvocationContext):
    if False:
        while True:
            i = 10
    helpers.set_api_id_stage_invocation_path(invocation_context)
    try:
        return invoke_rest_api(invocation_context)
    except AuthorizationError as e:
        LOG.warning('Authorization error while invoking API Gateway ID %s: %s', invocation_context.api_id, e, exc_info=LOG.isEnabledFor(logging.DEBUG))
        return e.to_response()

def invoke_rest_api(invocation_context: ApiInvocationContext):
    if False:
        while True:
            i = 10
    invocation_path = invocation_context.path_with_query_string
    raw_path = invocation_context.path or invocation_path
    method = invocation_context.method
    headers = invocation_context.headers
    (extracted_path, resource) = helpers.get_target_resource_details(invocation_context)
    if not resource:
        return make_error_response('Unable to find path %s' % invocation_context.path, 404)
    validator = RequestValidator(invocation_context)
    if not validator.is_request_valid():
        return make_error_response('Invalid request body', 400)
    api_key_required = resource.get('resourceMethods', {}).get(method, {}).get('apiKeyRequired')
    if api_key_required and (not is_api_key_valid(invocation_context)):
        raise AuthorizationError('Forbidden', 403)
    resource_methods = resource.get('resourceMethods', {})
    resource_method = resource_methods.get(method, {})
    if not resource_method:
        resource_method = resource_methods.get('ANY', {}) or resource_methods.get('X-AMAZON-APIGATEWAY-ANY-METHOD', {})
    method_integration = resource_method.get('methodIntegration')
    if not method_integration:
        if method == 'OPTIONS' and 'Origin' in headers:
            return get_cors_response(headers)
        return make_error_response('Unable to find integration for: %s %s (%s)' % (method, invocation_path, raw_path), 404)
    invocation_context.resource_path = extracted_path
    invocation_context.integration = method_integration
    return invoke_rest_api_integration(invocation_context)

def invoke_rest_api_integration(invocation_context: ApiInvocationContext):
    if False:
        while True:
            i = 10
    try:
        response = invoke_rest_api_integration_backend(invocation_context)
        invocation_context.response = response
        return response
    except ApiGatewayIntegrationError as e:
        LOG.warning('Error while invoking integration for ApiGateway ID %s: %s', invocation_context.api_id, e, exc_info=LOG.isEnabledFor(logging.DEBUG))
        return e.to_response()
    except Exception as e:
        msg = f"Error invoking integration for API Gateway ID '{invocation_context.api_id}': {e}"
        LOG.exception(msg)
        return make_error_response(msg, 400)

def invoke_rest_api_integration_backend(invocation_context: ApiInvocationContext):
    if False:
        return 10
    method = invocation_context.method
    headers = invocation_context.headers
    integration = invocation_context.integration
    integration_type_orig = integration.get('type') or integration.get('integrationType') or ''
    integration_type = integration_type_orig.upper()
    integration_method = integration.get('httpMethod')
    uri = integration.get('uri') or integration.get('integrationUri') or ''
    if uri.startswith('arn:aws:apigateway:') and ':lambda:path' in uri or uri.startswith('arn:aws:lambda'):
        if integration_type == 'AWS_PROXY':
            return LambdaProxyIntegration().invoke(invocation_context)
        elif integration_type == 'AWS':
            return LambdaIntegration().invoke(invocation_context)
    elif integration_type == 'AWS':
        if 'kinesis:action/' in uri:
            return KinesisIntegration().invoke(invocation_context)
        if 'states:action/' in uri:
            return StepFunctionIntegration().invoke(invocation_context)
        if ':dynamodb:action' in uri:
            return DynamoDBIntegration().invoke(invocation_context)
        if 's3:path/' in uri or 's3:action/' in uri:
            return S3Integration().invoke(invocation_context)
        if integration_method == 'POST' and ':sqs:path' in uri:
            return SQSIntegration().invoke(invocation_context)
        if method == 'POST' and ':sns:path' in uri:
            return SNSIntegration().invoke(invocation_context)
        if method == 'POST' and uri.startswith('arn:aws:apigateway:') and ('events:action/PutEvents' in uri):
            return EventBridgeIntegration().invoke(invocation_context)
    elif integration_type in ['HTTP_PROXY', 'HTTP']:
        return HTTPIntegration().invoke(invocation_context)
    elif integration_type == 'MOCK':
        return MockIntegration().invoke(invocation_context)
    if method == 'OPTIONS':
        return get_cors_response(headers)
    raise Exception(f'API Gateway integration type "{integration_type}", method "{method}", URI "{uri}" not yet implemented')