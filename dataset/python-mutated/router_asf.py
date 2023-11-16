import logging
from typing import Any, Dict
from requests.models import Response as RequestsResponse
from werkzeug.datastructures import Headers
from werkzeug.exceptions import NotFound
from localstack.constants import HEADER_LOCALSTACK_EDGE_URL
from localstack.http import Request, Response, Router
from localstack.http.dispatcher import Handler
from localstack.http.request import restore_payload
from localstack.services.apigateway.context import ApiInvocationContext
from localstack.services.apigateway.helpers import get_api_account_id_and_region
from localstack.services.apigateway.invocations import invoke_rest_api_from_request
from localstack.utils.aws.aws_responses import LambdaResponse
LOG = logging.getLogger(__name__)

def to_invocation_context(request: Request, url_params: Dict[str, Any]=None) -> ApiInvocationContext:
    if False:
        return 10
    '\n    Converts an HTTP Request object into an ApiInvocationContext.\n\n    :param request: the original request\n    :param url_params: the parameters extracted from the URL matching rules\n    :return: the ApiInvocationContext\n    '
    if url_params is None:
        url_params = {}
    method = request.method
    path = request.environ.get('RAW_URI')
    data = restore_payload(request)
    headers = Headers(request.headers)
    x_forwarded_for = headers.getlist('X-Forwarded-For')
    x_forwarded_for.append(request.remote_addr)
    x_forwarded_for.append(request.host)
    headers['X-Forwarded-For'] = ', '.join(x_forwarded_for)
    headers[HEADER_LOCALSTACK_EDGE_URL] = request.host_url.strip('/')
    return ApiInvocationContext(method, path, data, headers, stage=url_params.get('stage'))

def convert_response(result: RequestsResponse) -> Response:
    if False:
        print('Hello World!')
    '\n    Utility function to convert a response for the requests library to our internal (Werkzeug based) Response object.\n    '
    if result is None:
        return Response()
    if isinstance(result, LambdaResponse):
        headers = Headers(dict(result.headers))
        for (k, values) in result.multi_value_headers.items():
            for value in values:
                headers.add(k, value)
    else:
        headers = dict(result.headers)
    response = Response(status=result.status_code, headers=headers)
    if isinstance(result.content, dict):
        response.set_json(result.content)
    elif isinstance(result.content, (str, bytes)):
        response.data = result.content
    else:
        raise ValueError(f'Unhandled content type {type(result.content)}')
    return response

class ApigatewayRouter:
    """
    Simple implementation around a Router to manage dynamic restapi routes (routes added by a user through the
    apigateway API).
    """
    router: Router[Handler]

    def __init__(self, router: Router[Handler]):
        if False:
            print('Hello World!')
        self.router = router
        self.registered = False

    def register_routes(self) -> None:
        if False:
            while True:
                i = 10
        'Registers parameterized routes for API Gateway user invocations.'
        if self.registered:
            LOG.debug('Skipped API Gateway route registration (routes already registered).')
            return
        self.registered = True
        LOG.debug('Registering parameterized API Gateway routes.')
        host_pattern = "<regex('[^-]+'):api_id><regex('(-vpce-[^.]+)?'):vpce_suffix>.execute-api.<regex('.*'):server>"
        self.router.add('/', host=host_pattern, endpoint=self.invoke_rest_api, defaults={'path': '', 'stage': None}, strict_slashes=True)
        self.router.add('/<stage>/', host=host_pattern, endpoint=self.invoke_rest_api, defaults={'path': ''}, strict_slashes=False)
        self.router.add('/<stage>/<path:path>', host=host_pattern, endpoint=self.invoke_rest_api, strict_slashes=True)
        self.router.add('/restapis/<api_id>/<stage>/_user_request_', endpoint=self.invoke_rest_api, defaults={'path': ''})
        self.router.add('/restapis/<api_id>/<stage>/_user_request_/<path:path>', endpoint=self.invoke_rest_api, strict_slashes=True)

    def invoke_rest_api(self, request: Request, **url_params: Dict[str, str]) -> Response:
        if False:
            for i in range(10):
                print('nop')
        (account_id, region_name) = get_api_account_id_and_region(url_params['api_id'])
        if not region_name:
            return Response(status=404)
        invocation_context = to_invocation_context(request, url_params)
        invocation_context.region_name = region_name
        invocation_context.account_id = account_id
        result = invoke_rest_api_from_request(invocation_context)
        if result is not None:
            return convert_response(result)
        raise NotFound()