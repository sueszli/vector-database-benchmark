"""API Gateway Local Service"""
import base64
import json
import logging
from datetime import datetime
from io import StringIO
from time import time
from typing import Any, Dict, List, Optional
from flask import Flask, Request, request
from werkzeug.datastructures import Headers
from werkzeug.routing import BaseConverter
from werkzeug.serving import WSGIRequestHandler
from samcli.commands.local.lib.exceptions import UnsupportedInlineCodeError
from samcli.commands.local.lib.local_lambda import LocalLambdaRunner
from samcli.lib.providers.provider import Api, Cors
from samcli.lib.telemetry.event import EventName, EventTracker, UsedFeature
from samcli.lib.utils.stream_writer import StreamWriter
from samcli.local.apigw.authorizers.lambda_authorizer import LambdaAuthorizer
from samcli.local.apigw.event_constructor import construct_v1_event, construct_v2_event_http
from samcli.local.apigw.exceptions import AuthorizerUnauthorizedRequest, InvalidLambdaAuthorizerResponse, InvalidSecurityDefinition, LambdaResponseParseException, PayloadFormatVersionValidateException
from samcli.local.apigw.path_converter import PathConverter
from samcli.local.apigw.route import Route
from samcli.local.apigw.service_error_responses import ServiceErrorResponses
from samcli.local.events.api_event import ContextHTTP, ContextIdentity, RequestContext, RequestContextV2
from samcli.local.lambdafn.exceptions import FunctionNotFound
from samcli.local.services.base_local_service import BaseLocalService, LambdaOutputParser
LOG = logging.getLogger(__name__)

class CatchAllPathConverter(BaseConverter):
    regex = '.+'
    weight = 300
    part_isolating = False

    def to_python(self, value):
        if False:
            i = 10
            return i + 15
        return value

    def to_url(self, value):
        if False:
            return 10
        return value

class LocalApigwService(BaseLocalService):
    _DEFAULT_PORT = 3000
    _DEFAULT_HOST = '127.0.0.1'

    def __init__(self, api: Api, lambda_runner: LocalLambdaRunner, static_dir: Optional[str]=None, port: Optional[int]=None, host: Optional[str]=None, stderr: Optional[StreamWriter]=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Creates an ApiGatewayService\n\n        Parameters\n        ----------\n        api : Api\n           an Api object that contains the list of routes and properties\n        lambda_runner : samcli.commands.local.lib.local_lambda.LocalLambdaRunner\n            The Lambda runner class capable of invoking the function\n        static_dir : str\n            Directory from which to serve static files\n        port : int\n            Optional. port for the service to start listening on\n            Defaults to 3000\n        host : str\n            Optional. host to start the service on\n            Defaults to '127.0.0.1\n        stderr : samcli.lib.utils.stream_writer.StreamWriter\n            Optional stream writer where the stderr from Docker container should be written to\n        "
        super().__init__(lambda_runner.is_debugging(), port=port, host=host)
        self.api = api
        self.lambda_runner = lambda_runner
        self.static_dir = static_dir
        self._dict_of_routes: Dict[str, Route] = {}
        self.stderr = stderr
        self._click_session_id = None
        try:
            from samcli.cli.context import Context
            ctx = Context.get_current_context()
            if ctx:
                self._click_session_id = ctx.session_id
        except RuntimeError:
            LOG.debug('Not able to get click context in APIGW service')

    def create(self):
        if False:
            print('Hello World!')
        '\n        Creates a Flask Application that can be started.\n        '
        WSGIRequestHandler.protocol_version = 'HTTP/1.1'
        self._app = Flask(__name__, static_url_path='', static_folder=self.static_dir)
        self._app.url_map.converters['path'] = CatchAllPathConverter
        self._app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
        self._app.url_map.strict_slashes = False
        default_route = None
        for api_gateway_route in self.api.routes:
            if api_gateway_route.path == '$default':
                default_route = api_gateway_route
                continue
            path = PathConverter.convert_path_to_flask(api_gateway_route.path)
            for route_key in self._generate_route_keys(api_gateway_route.methods, path):
                self._dict_of_routes[route_key] = api_gateway_route
            self._app.add_url_rule(path, endpoint=path, view_func=self._request_handler, methods=api_gateway_route.methods, provide_automatic_options=False)
        if default_route:
            LOG.debug('add catch-all route')
            all_methods = Route.ANY_HTTP_METHODS
            try:
                rules_iter = self._app.url_map.iter_rules('/')
                while True:
                    rule = next(rules_iter)
                    all_methods = [method for method in all_methods if method not in rule.methods]
            except (KeyError, StopIteration):
                pass
            self._add_catch_all_path(all_methods, '/', default_route)
            self._add_catch_all_path(Route.ANY_HTTP_METHODS, '/<path:any_path>', default_route)
        self._construct_error_handling()

    def _add_catch_all_path(self, methods: List[str], path: str, route: Route):
        if False:
            while True:
                i = 10
        '\n        Add the catch all route to the _app and the dictionary of routes.\n\n        :param list(str) methods: List of HTTP Methods\n        :param str path: Path off the base url\n        :param Route route: contains the default route configurations\n        '
        self._app.add_url_rule(path, endpoint=path, view_func=self._request_handler, methods=methods, provide_automatic_options=False)
        for route_key in self._generate_route_keys(methods, path):
            self._dict_of_routes[route_key] = Route(function_name=route.function_name, path=path, methods=methods, event_type=Route.HTTP, payload_format_version=route.payload_format_version, is_default_route=True, stack_path=route.stack_path, authorizer_name=route.authorizer_name, authorizer_object=route.authorizer_object, use_default_authorizer=route.use_default_authorizer)

    def _generate_route_keys(self, methods, path):
        if False:
            print('Hello World!')
        '\n        Generates the key to the _dict_of_routes based on the list of methods\n        and path supplied\n\n        Parameters\n        ----------\n        methods : List[str]\n            List of HTTP Methods\n        path : str\n            Path off the base url\n\n        Yields\n        ------\n        route_key : str\n            the route key in the form of "Path:Method"\n        '
        for method in methods:
            yield self._route_key(method, path)

    @staticmethod
    def _v2_route_key(method, path, is_default_route):
        if False:
            while True:
                i = 10
        if is_default_route:
            return '$default'
        return '{} {}'.format(method, path)

    @staticmethod
    def _route_key(method, path):
        if False:
            return 10
        return '{}:{}'.format(path, method)

    def _construct_error_handling(self):
        if False:
            return 10
        '\n        Updates the Flask app with Error Handlers for different Error Codes\n        '
        self._app.register_error_handler(404, ServiceErrorResponses.route_not_found)
        self._app.register_error_handler(405, ServiceErrorResponses.route_not_found)
        self._app.register_error_handler(500, ServiceErrorResponses.lambda_failure_response)

    def _create_method_arn(self, flask_request: Request, event_type: str) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Creates a method ARN with fake AWS values\n\n        Parameters\n        ----------\n        flask_request: Request\n            Flask request object to get method and path\n        event_type: str\n            Type of event (API or HTTP)\n\n        Returns\n        -------\n        str\n            A built method ARN with fake values\n        '
        context = RequestContext() if event_type == Route.API else RequestContextV2()
        (method, path) = (flask_request.method, flask_request.path)
        return f'arn:aws:execute-api:us-east-1:{context.account_id}:{context.api_id}/{self.api.stage_name}/{method}{path}'

    def _generate_lambda_token_authorizer_event(self, flask_request: Request, route: Route, lambda_authorizer: LambdaAuthorizer) -> dict:
        if False:
            while True:
                i = 10
        '\n        Creates a Lambda authorizer token event\n\n        Parameters\n        ----------\n        flask_request: Request\n            Flask request object to get method and path\n        route: Route\n            Route object representing the endpoint to be invoked later\n        lambda_authorizer: LambdaAuthorizer\n            The Lambda authorizer the route is using\n\n        Returns\n        -------\n        dict\n            Basic dictionary containing a type and authorizationToken\n        '
        method_arn = self._create_method_arn(flask_request, route.event_type)
        headers = {'headers': flask_request.headers}
        if len(lambda_authorizer.identity_sources) != 1:
            raise InvalidSecurityDefinition('An invalid token based Lambda Authorizer was found, there should be one header identity source')
        identity_source = lambda_authorizer.identity_sources[0]
        authorization_token = identity_source.find_identity_value(**headers)
        return {'type': LambdaAuthorizer.TOKEN.upper(), 'authorizationToken': str(authorization_token), 'methodArn': method_arn}

    def _generate_lambda_request_authorizer_event_http(self, lambda_authorizer_payload: str, identity_values: list, method_arn: str) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Helper method to generate part of the event required for different payload versions\n        for API Gateway V2\n\n        Parameters\n        ----------\n        lambda_authorizer_payload: str\n            The payload version of the Lambda authorizer\n        identity_values: list\n            A list of string identity values\n        method_arn: str\n            The method ARN for the endpoint\n\n        Returns\n        -------\n        dict\n            Dictionary containing partial Lambda authorizer event\n        '
        if lambda_authorizer_payload == LambdaAuthorizer.PAYLOAD_V2:
            return {'identitySource': identity_values, 'routeArn': method_arn}
        else:
            all_identity_values_string = ','.join(identity_values)
            return {'identitySource': all_identity_values_string, 'authorizationToken': all_identity_values_string, 'methodArn': method_arn}

    def _generate_lambda_request_authorizer_event(self, flask_request: Request, route: Route, lambda_authorizer: LambdaAuthorizer) -> dict:
        if False:
            print('Hello World!')
        '\n        Creates a Lambda authorizer request event\n\n        Parameters\n        ----------\n        flask_request: Request\n            Flask request object to get method and path\n        route: Route\n            Route object representing the endpoint to be invoked later\n        lambda_authorizer: LambdaAuthorizer\n            The Lambda authorizer the route is using\n\n        Returns\n        -------\n        dict\n            A Lambda authorizer event\n        '
        method_arn = self._create_method_arn(flask_request, route.event_type)
        (method, endpoint) = self.get_request_methods_endpoints(flask_request)
        lambda_event = self._generate_lambda_event(flask_request, route, method, endpoint)
        lambda_event.update({'type': LambdaAuthorizer.REQUEST.upper()})
        context = self._build_v1_context(route) if lambda_authorizer.payload_version == LambdaAuthorizer.PAYLOAD_V1 else self._build_v2_context(route)
        if route.event_type == Route.API:
            lambda_event.update({'methodArn': method_arn})
        else:
            kwargs = {'headers': flask_request.headers, 'querystring': flask_request.query_string.decode('utf-8'), 'context': context, 'stageVariables': self.api.stage_variables}
            all_identity_values = []
            for identity_source in lambda_authorizer.identity_sources:
                value = identity_source.find_identity_value(**kwargs)
                if value:
                    all_identity_values.append(str(value))
            lambda_event.update(self._generate_lambda_request_authorizer_event_http(lambda_authorizer.payload_version, all_identity_values, method_arn))
        return lambda_event

    def _generate_lambda_authorizer_event(self, flask_request: Request, route: Route, lambda_authorizer: LambdaAuthorizer) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Generate a Lambda authorizer event\n\n        Parameters\n        ----------\n        flask_request: Request\n            Flask request object to get method and endpoint\n        route: Route\n            Route object representing the endpoint to be invoked later\n        lambda_authorizer: LambdaAuthorizer\n            The Lambda authorizer the route is using\n\n        Returns\n        -------\n        str\n            A JSON string containing event properties\n        '
        authorizer_events = {LambdaAuthorizer.TOKEN: self._generate_lambda_token_authorizer_event, LambdaAuthorizer.REQUEST: self._generate_lambda_request_authorizer_event}
        kwargs: Dict[str, Any] = {'flask_request': flask_request, 'route': route, 'lambda_authorizer': lambda_authorizer}
        return authorizer_events[lambda_authorizer.type](**kwargs)

    def _generate_lambda_event(self, flask_request: Request, route: Route, method: str, endpoint: str) -> dict:
        if False:
            return 10
        '\n        Helper function to generate the correct Lambda event\n\n        Parameters\n        ----------\n        flask_request: Request\n            The global Flask Request object\n        route: Route\n            The Route that was called\n        method: str\n            The method of the request (eg. GET, POST) from the Flask request\n        endpoint: str\n            The endpoint of the request from the Flask request\n\n        Returns\n        -------\n        str\n            JSON string of event properties\n        '
        if route.event_type == Route.HTTP and route.payload_format_version in [None, '2.0']:
            apigw_endpoint = PathConverter.convert_path_to_api_gateway(endpoint)
            route_key = self._v2_route_key(method, apigw_endpoint, route.is_default_route)
            return construct_v2_event_http(flask_request=flask_request, port=self.port, binary_types=self.api.binary_media_types, stage_name=self.api.stage_name, stage_variables=self.api.stage_variables, route_key=route_key)
        route_key = route.operation_name if route.event_type == Route.API else None
        return construct_v1_event(flask_request=flask_request, port=self.port, binary_types=self.api.binary_media_types, stage_name=self.api.stage_name, stage_variables=self.api.stage_variables, operation_name=route_key)

    def _build_v1_context(self, route: Route) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        '\n        Helper function to a 1.0 request context\n\n        Parameters\n        ----------\n        route: Route\n            The Route object that was invoked\n\n        Returns\n        -------\n        dict\n            JSON object containing context variables\n        '
        identity = ContextIdentity(source_ip=request.remote_addr)
        protocol = request.environ.get('SERVER_PROTOCOL', 'HTTP/1.1')
        host = request.host
        operation_name = route.operation_name if route.event_type == Route.API else None
        endpoint = PathConverter.convert_path_to_api_gateway(request.endpoint)
        method = request.method
        context = RequestContext(resource_path=endpoint, http_method=method, stage=self.api.stage_name, identity=identity, path=endpoint, protocol=protocol, domain_name=host, operation_name=operation_name)
        return context.to_dict()

    def _build_v2_context(self, route: Route) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        '\n        Helper function to a 2.0 request context\n\n        Parameters\n        ----------\n        route: Route\n            The Route object that was invoked\n\n        Returns\n        -------\n        dict\n            JSON object containing context variables\n        '
        endpoint = PathConverter.convert_path_to_api_gateway(request.endpoint)
        method = request.method
        apigw_endpoint = PathConverter.convert_path_to_api_gateway(endpoint)
        route_key = self._v2_route_key(method, apigw_endpoint, route.is_default_route)
        request_time_epoch = int(time())
        request_time = datetime.utcnow().strftime('%d/%b/%Y:%H:%M:%S +0000')
        context_http = ContextHTTP(method=method, path=request.path, source_ip=request.remote_addr)
        context = RequestContextV2(http=context_http, route_key=route_key, stage=self.api.stage_name, request_time_epoch=request_time_epoch, request_time=request_time)
        return context.to_dict()

    def _valid_identity_sources(self, request: Request, route: Route) -> bool:
        if False:
            for i in range(10):
                print('nop')
        "\n        Validates if the route contains all the valid identity sources defined in the route's Lambda Authorizer\n\n        Parameters\n        ----------\n        request: Request\n            Flask request object containing incoming request variables\n        route: Route\n            the Route object that contains the Lambda Authorizer definition\n\n        Returns\n        -------\n        bool\n            true if all the identity sources are present and valid\n        "
        lambda_auth = route.authorizer_object
        if not isinstance(lambda_auth, LambdaAuthorizer):
            return False
        identity_sources = lambda_auth.identity_sources
        context = self._build_v1_context(route) if lambda_auth.payload_version == LambdaAuthorizer.PAYLOAD_V1 else self._build_v2_context(route)
        kwargs = {'headers': request.headers, 'querystring': request.query_string.decode('utf-8'), 'context': context, 'stageVariables': self.api.stage_variables, 'validation_expression': lambda_auth.validation_string}
        for validator in identity_sources:
            if not validator.is_valid(**kwargs):
                return False
        return True

    def _invoke_lambda_function(self, lambda_function_name: str, event: dict) -> str:
        if False:
            return 10
        '\n        Helper method to invoke a function and setup stdout+stderr\n\n        Parameters\n        ----------\n        lambda_function_name: str\n            The name of the Lambda function to invoke\n        event: dict\n            The event object to pass into the Lambda function\n\n        Returns\n        -------\n        str\n            A string containing the output from the Lambda function\n        '
        with StringIO() as stdout:
            event_str = json.dumps(event, sort_keys=True)
            stdout_writer = StreamWriter(stdout, auto_flush=True)
            self.lambda_runner.invoke(lambda_function_name, event_str, stdout=stdout_writer, stderr=self.stderr)
            (lambda_response, is_lambda_user_error_response) = LambdaOutputParser.get_lambda_output(stdout)
            if is_lambda_user_error_response:
                raise LambdaResponseParseException
        return lambda_response

    def _request_handler(self, **kwargs):
        if False:
            return 10
        "\n        We handle all requests to the host:port. The general flow of handling a request is as follows\n\n        * Fetch request from the Flask Global state. This is where Flask places the request and is per thread so\n          multiple requests are still handled correctly\n        * Find the Lambda function to invoke by doing a look up based on the request.endpoint and method\n        * If we don't find the function, we will throw a 502 (just like the 404 and 405 responses we get\n          from Flask.\n        * Since we found a Lambda function to invoke, we construct the Lambda Event from the request\n        * Then Invoke the Lambda function (docker container)\n        * We then transform the response or errors we get from the Invoke and return the data back to\n          the caller\n\n        Parameters\n        ----------\n        kwargs dict\n            Keyword Args that are passed to the function from Flask. This happens when we have path parameters\n\n        Returns\n        -------\n        Response object\n        "
        route: Route = self._get_current_route(request)
        request_origin = request.headers.get('Origin')
        cors_headers = Cors.cors_to_headers(self.api.cors, request_origin, route.event_type)
        lambda_authorizer = route.authorizer_object
        if route.payload_format_version not in [None, '1.0', '2.0']:
            raise PayloadFormatVersionValidateException(f'{route.payload_format_version} is not a valid value. PayloadFormatVersion must be "1.0" or "2.0"')
        (method, endpoint) = self.get_request_methods_endpoints(request)
        if method == 'OPTIONS' and self.api.cors:
            headers = Headers(cors_headers)
            return self.service_response('', headers, 200)
        if isinstance(lambda_authorizer, LambdaAuthorizer) and (not self._valid_identity_sources(request, route)):
            return ServiceErrorResponses.missing_lambda_auth_identity_sources()
        try:
            route_lambda_event = self._generate_lambda_event(request, route, method, endpoint)
            auth_lambda_event = None
            if lambda_authorizer:
                auth_lambda_event = self._generate_lambda_authorizer_event(request, route, lambda_authorizer)
        except UnicodeDecodeError as error:
            LOG.error('UnicodeDecodeError while processing HTTP request: %s', error)
            return ServiceErrorResponses.lambda_failure_response()
        lambda_authorizer_exception = None
        try:
            auth_service_error = None
            if lambda_authorizer:
                self._invoke_parse_lambda_authorizer(lambda_authorizer, auth_lambda_event, route_lambda_event, route)
        except AuthorizerUnauthorizedRequest as ex:
            auth_service_error = ServiceErrorResponses.lambda_authorizer_unauthorized()
            lambda_authorizer_exception = ex
        except InvalidLambdaAuthorizerResponse as ex:
            auth_service_error = ServiceErrorResponses.lambda_failure_response()
            lambda_authorizer_exception = ex
        except FunctionNotFound as ex:
            lambda_authorizer_exception = ex
            LOG.warning('Failed to find a Function to invoke a Lambda authorizer, verify that this Function is defined and exists locally in the template.')
        except Exception as ex:
            lambda_authorizer_exception = ex
            raise ex
        finally:
            exception_name = type(lambda_authorizer_exception).__name__ if lambda_authorizer_exception else None
            EventTracker.track_event(event_name=EventName.USED_FEATURE.value, event_value=UsedFeature.INVOKED_CUSTOM_LAMBDA_AUTHORIZERS.value, session_id=self._click_session_id, exception_name=exception_name)
            if lambda_authorizer_exception:
                LOG.error('Lambda authorizer failed to invoke successfully: %s', str(lambda_authorizer_exception))
            if auth_service_error:
                return auth_service_error
        endpoint_service_error = None
        try:
            lambda_response = self._invoke_lambda_function(route.function_name, route_lambda_event)
        except FunctionNotFound:
            endpoint_service_error = ServiceErrorResponses.lambda_not_found_response()
        except UnsupportedInlineCodeError:
            endpoint_service_error = ServiceErrorResponses.not_implemented_locally('Inline code is not supported for sam local commands. Please write your code in a separate file.')
        except LambdaResponseParseException:
            endpoint_service_error = ServiceErrorResponses.lambda_body_failure_response()
        if endpoint_service_error:
            return endpoint_service_error
        try:
            if route.event_type == Route.HTTP and (not route.payload_format_version or route.payload_format_version == '2.0'):
                (status_code, headers, body) = self._parse_v2_payload_format_lambda_output(lambda_response, self.api.binary_media_types, request)
            else:
                (status_code, headers, body) = self._parse_v1_payload_format_lambda_output(lambda_response, self.api.binary_media_types, request, route.event_type)
        except LambdaResponseParseException as ex:
            LOG.error('Invalid lambda response received: %s', ex)
            return ServiceErrorResponses.lambda_failure_response()
        headers.update(cors_headers)
        return self.service_response(body, headers, status_code)

    def _invoke_parse_lambda_authorizer(self, lambda_authorizer: LambdaAuthorizer, auth_lambda_event: dict, route_lambda_event: dict, route: Route) -> None:
        if False:
            return 10
        "\n        Helper method to invoke and parse the output of a Lambda authorizer\n\n        Parameters\n        ----------\n        lambda_authorizer: LambdaAuthorizer\n            The route's Lambda authorizer\n        auth_lambda_event: dict\n            The event to pass to the Lambda authorizer\n        route_lambda_event: dict\n            The event to pass into the route\n        route: Route\n            The route that is being called\n        "
        lambda_auth_response = self._invoke_lambda_function(lambda_authorizer.lambda_name, auth_lambda_event)
        method_arn = self._create_method_arn(request, route.event_type)
        if not lambda_authorizer.is_valid_response(lambda_auth_response, method_arn):
            raise AuthorizerUnauthorizedRequest(f'Request is not authorized for {method_arn}')
        original_context = route_lambda_event.get('requestContext', {})
        context = lambda_authorizer.get_context(lambda_auth_response)
        if route.event_type == Route.HTTP and route.payload_format_version in [None, '2.0']:
            original_context.update({'authorizer': {'lambda': context}})
        else:
            original_context.update({'authorizer': context})
        route_lambda_event.update({'requestContext': original_context})

    def _get_current_route(self, flask_request):
        if False:
            print('Hello World!')
        '\n        Get the route (Route) based on the current request\n\n        :param request flask_request: Flask Request\n        :return: Route matching the endpoint and method of the request\n        '
        (method, endpoint) = self.get_request_methods_endpoints(flask_request)
        route_key = self._route_key(method, endpoint)
        route = self._dict_of_routes.get(route_key, None)
        if not route:
            LOG.debug('Lambda function for the route not found. This should not happen because Flask is already configured to serve all path/methods given to the service. Path=%s Method=%s RouteKey=%s', endpoint, method, route_key)
            raise KeyError('Lambda function for the route not found')
        return route

    @staticmethod
    def get_request_methods_endpoints(flask_request):
        if False:
            return 10
        "\n        Separated out for testing requests in request handler\n        :param request flask_request: Flask Request\n        :return: the request's endpoint and method\n        "
        endpoint = flask_request.endpoint
        method = flask_request.method
        return (method, endpoint)

    @staticmethod
    def _parse_v1_payload_format_lambda_output(lambda_output: str, binary_types, flask_request, event_type):
        if False:
            i = 10
            return i + 15
        '\n        Parses the output from the Lambda Container\n\n        :param str lambda_output: Output from Lambda Invoke\n        :param binary_types: list of binary types\n        :param flask_request: flash request object\n        :param event_type: determines the route event type\n        :return: Tuple(int, dict, str, bool)\n        '
        try:
            json_output = json.loads(lambda_output)
        except ValueError as ex:
            raise LambdaResponseParseException('Lambda response must be valid json') from ex
        if not isinstance(json_output, dict):
            raise LambdaResponseParseException(f'Lambda returned {type(json_output)} instead of dict')
        if event_type == Route.HTTP and json_output.get('statusCode') is None:
            raise LambdaResponseParseException(f'Invalid API Gateway Response Key: statusCode is not in {json_output}')
        status_code = json_output.get('statusCode') or 200
        headers = LocalApigwService._merge_response_headers(json_output.get('headers') or {}, json_output.get('multiValueHeaders') or {})
        body = json_output.get('body')
        if body is None:
            LOG.warning('Lambda returned empty body!')
        is_base_64_encoded = LocalApigwService.get_base_64_encoded(event_type, json_output)
        try:
            status_code = int(status_code)
            if status_code <= 0:
                raise ValueError
        except ValueError as ex:
            raise LambdaResponseParseException('statusCode must be a positive int') from ex
        try:
            if body:
                body = str(body)
        except ValueError as ex:
            raise LambdaResponseParseException(f'Non null response bodies should be able to convert to string: {body}') from ex
        invalid_keys = LocalApigwService._invalid_apig_response_keys(json_output, event_type)
        if event_type == Route.API and invalid_keys:
            raise LambdaResponseParseException(f'Invalid API Gateway Response Keys: {invalid_keys} in {json_output}')
        if 'Content-Type' not in headers:
            LOG.info("No Content-Type given. Defaulting to 'application/json'.")
            headers['Content-Type'] = 'application/json'
        try:
            if event_type == Route.HTTP and is_base_64_encoded or (event_type == Route.API and LocalApigwService._should_base64_decode_body(binary_types, flask_request, headers, is_base_64_encoded)):
                body = base64.b64decode(body)
        except ValueError as ex:
            LambdaResponseParseException(str(ex))
        return (status_code, headers, body)

    @staticmethod
    def get_base_64_encoded(event_type, json_output):
        if False:
            for i in range(10):
                print('nop')
        if event_type == Route.API and json_output.get('base64Encoded') is not None:
            is_base_64_encoded = json_output.get('base64Encoded')
            field_name = 'base64Encoded'
        elif json_output.get('isBase64Encoded') is not None:
            is_base_64_encoded = json_output.get('isBase64Encoded')
            field_name = 'isBase64Encoded'
        else:
            is_base_64_encoded = False
            field_name = 'isBase64Encoded'
        if isinstance(is_base_64_encoded, str) and is_base_64_encoded in ['true', 'True', 'false', 'False']:
            is_base_64_encoded = is_base_64_encoded in ['true', 'True']
        elif not isinstance(is_base_64_encoded, bool):
            raise LambdaResponseParseException(f'Invalid API Gateway Response Key: {is_base_64_encoded} is not a valid{field_name}')
        return is_base_64_encoded

    @staticmethod
    def _parse_v2_payload_format_lambda_output(lambda_output: str, binary_types, flask_request):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parses the output from the Lambda Container. V2 Payload Format means that the event_type is only HTTP\n\n        :param str lambda_output: Output from Lambda Invoke\n        :param binary_types: list of binary types\n        :param flask_request: flash request object\n        :return: Tuple(int, dict, str, bool)\n        '
        try:
            json_output = json.loads(lambda_output)
        except ValueError as ex:
            raise LambdaResponseParseException('Lambda response must be valid json') from ex
        if isinstance(json_output, dict):
            body = json_output.get('body') if 'statusCode' in json_output else json.dumps(json_output)
        else:
            body = json_output
            json_output = {}
        if body is None:
            LOG.warning('Lambda returned empty body!')
        status_code = json_output.get('statusCode') or 200
        headers = Headers(json_output.get('headers') or {})
        cookies = json_output.get('cookies')
        if isinstance(cookies, list):
            for cookie in cookies:
                headers.add('Set-Cookie', cookie)
        is_base_64_encoded = json_output.get('isBase64Encoded') or False
        try:
            status_code = int(status_code)
            if status_code <= 0:
                raise ValueError
        except ValueError as ex:
            raise LambdaResponseParseException('statusCode must be a positive int') from ex
        try:
            if body:
                body = str(body)
        except ValueError as ex:
            raise LambdaResponseParseException(f'Non null response bodies should be able to convert to string: {body}') from ex
        if 'Content-Type' not in headers:
            LOG.info("No Content-Type given. Defaulting to 'application/json'.")
            headers['Content-Type'] = 'application/json'
        try:
            if is_base_64_encoded:
                body = base64.b64decode(body)
        except ValueError as ex:
            LambdaResponseParseException(str(ex))
        return (status_code, headers, body)

    @staticmethod
    def _invalid_apig_response_keys(output, event_type):
        if False:
            for i in range(10):
                print('nop')
        allowable = {'statusCode', 'body', 'headers', 'multiValueHeaders', 'isBase64Encoded', 'cookies'}
        if event_type == Route.API:
            allowable.add('base64Encoded')
        invalid_keys = output.keys() - allowable
        return invalid_keys

    @staticmethod
    def _should_base64_decode_body(binary_types, flask_request, lamba_response_headers, is_base_64_encoded):
        if False:
            while True:
                i = 10
        '\n        Whether or not the body should be decoded from Base64 to Binary\n\n        Parameters\n        ----------\n        binary_types list(basestring)\n            Corresponds to self.binary_types (aka. what is parsed from SAM Template\n        flask_request flask.request\n            Flask request\n        lamba_response_headers werkzeug.datastructures.Headers\n            Headers Lambda returns\n        is_base_64_encoded bool\n            True if the body is Base64 encoded\n\n        Returns\n        -------\n        True if the body from the request should be converted to binary, otherwise false\n\n        '
        best_match_mimetype = flask_request.accept_mimetypes.best_match(lamba_response_headers.get_all('Content-Type'))
        is_best_match_in_binary_types = best_match_mimetype in binary_types or '*/*' in binary_types
        return best_match_mimetype and is_best_match_in_binary_types and is_base_64_encoded

    @staticmethod
    def _merge_response_headers(headers, multi_headers):
        if False:
            return 10
        '\n        Merge multiValueHeaders headers with headers\n\n        * If you specify values for both headers and multiValueHeaders, API Gateway merges them into a single list.\n        * If the same key-value pair is specified in both, the value will only appear once.\n\n        Parameters\n        ----------\n        headers dict\n            Headers map from the lambda_response_headers\n        multi_headers dict\n            multiValueHeaders map from the lambda_response_headers\n\n        Returns\n        -------\n        Merged list in accordance to the AWS documentation within a Flask Headers object\n\n        '
        processed_headers = Headers(multi_headers)
        for header in headers:
            if header in multi_headers and headers[header] in multi_headers[header]:
                continue
            processed_headers.add(header, headers[header])
        return processed_headers