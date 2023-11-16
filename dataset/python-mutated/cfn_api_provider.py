"""Parses SAM given a template"""
import logging
from typing import Any, Dict, List, Optional, Tuple, cast
from samcli.commands.local.cli_common.user_exceptions import InvalidSamTemplateException
from samcli.commands.local.lib.swagger.integration_uri import LambdaUri
from samcli.commands.local.lib.validators.identity_source_validator import IdentitySourceValidator
from samcli.commands.local.lib.validators.lambda_auth_props import LambdaAuthorizerV1Validator, LambdaAuthorizerV2Validator
from samcli.lib.providers.api_collector import ApiCollector
from samcli.lib.providers.cfn_base_api_provider import CfnBaseApiProvider
from samcli.lib.providers.provider import Stack
from samcli.lib.utils.resources import AWS_APIGATEWAY_AUTHORIZER, AWS_APIGATEWAY_METHOD, AWS_APIGATEWAY_RESOURCE, AWS_APIGATEWAY_RESTAPI, AWS_APIGATEWAY_STAGE, AWS_APIGATEWAY_V2_API, AWS_APIGATEWAY_V2_AUTHORIZER, AWS_APIGATEWAY_V2_INTEGRATION, AWS_APIGATEWAY_V2_ROUTE, AWS_APIGATEWAY_V2_STAGE
from samcli.local.apigw.authorizers.lambda_authorizer import LambdaAuthorizer
from samcli.local.apigw.route import Route
LOG = logging.getLogger(__name__)

class CfnApiProvider(CfnBaseApiProvider):
    METHOD_BINARY_TYPE = 'CONVERT_TO_BINARY'
    HTTP_API_PROTOCOL_TYPE = 'HTTP'
    TYPES = [AWS_APIGATEWAY_RESTAPI, AWS_APIGATEWAY_STAGE, AWS_APIGATEWAY_RESOURCE, AWS_APIGATEWAY_METHOD, AWS_APIGATEWAY_AUTHORIZER, AWS_APIGATEWAY_V2_API, AWS_APIGATEWAY_V2_INTEGRATION, AWS_APIGATEWAY_V2_ROUTE, AWS_APIGATEWAY_V2_STAGE, AWS_APIGATEWAY_V2_AUTHORIZER]
    _METHOD_AUTHORIZER_ID = 'AuthorizerId'
    _ROUTE_AUTHORIZER_ID = 'AuthorizerId'

    def extract_resources(self, stacks: List[Stack], collector: ApiCollector, cwd: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        '\n        Extract the Route Object from a given resource and adds it to the RouteCollector.\n\n        Parameters\n        ----------\n        stacks: List[Stack]\n            List of stacks apis are extracted from\n\n        collector: samcli.lib.providers.api_collector.ApiCollector\n            Instance of the API collector that where we will save the API information\n\n        cwd : str\n            Optional working directory with respect to which we will resolve relative path to Swagger file\n        '
        for stack in stacks:
            resources = stack.resources
            for (logical_id, resource) in resources.items():
                resource_type = resource.get(CfnBaseApiProvider.RESOURCE_TYPE)
                if resource_type == AWS_APIGATEWAY_RESTAPI:
                    self._extract_cloud_formation_route(stack.stack_path, logical_id, resource, collector, cwd=cwd)
                if resource_type == AWS_APIGATEWAY_STAGE:
                    self._extract_cloud_formation_stage(resources, resource, collector)
                if resource_type == AWS_APIGATEWAY_METHOD:
                    self._extract_cloud_formation_method(stack.stack_path, resources, logical_id, resource, collector)
                if resource_type == AWS_APIGATEWAY_AUTHORIZER:
                    self._extract_cloud_formation_authorizer(logical_id, resource, collector)
                if resource_type == AWS_APIGATEWAY_V2_API:
                    self._extract_cfn_gateway_v2_api(stack.stack_path, logical_id, resource, collector, cwd=cwd)
                if resource_type == AWS_APIGATEWAY_V2_ROUTE:
                    self._extract_cfn_gateway_v2_route(stack.stack_path, resources, logical_id, resource, collector)
                if resource_type == AWS_APIGATEWAY_V2_STAGE:
                    self._extract_cfn_gateway_v2_stage(resources, resource, collector)
                if resource_type == AWS_APIGATEWAY_V2_AUTHORIZER:
                    self._extract_cfn_gateway_v2_authorizer(logical_id, resource, collector)

    @staticmethod
    def _extract_cloud_formation_authorizer(logical_id: str, resource: dict, collector: ApiCollector) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Extract Authorizers from AWS::ApiGateway::Authorizer and add them to the collector.\n\n        Parameters\n        ----------\n        logical_id: str\n            The logical ID of the Authorizer\n        resource: dict\n            The attributes for the Authorizer\n        collector: ApiCollector\n            ApiCollector to save Authorizers into\n        '
        if not LambdaAuthorizerV1Validator.validate(logical_id, resource):
            return
        properties = resource.get('Properties', {})
        authorizer_type = properties.get(LambdaAuthorizerV1Validator.AUTHORIZER_TYPE, '').lower()
        rest_api_id = properties.get(LambdaAuthorizerV1Validator.AUTHORIZER_REST_API)
        name = properties.get(LambdaAuthorizerV1Validator.AUTHORIZER_NAME)
        authorizer_uri = properties.get(LambdaAuthorizerV1Validator.AUTHORIZER_AUTHORIZER_URI)
        identity_source_template = properties.get(LambdaAuthorizerV1Validator.AUTHORIZER_IDENTITY_SOURCE, '')
        function_name = cast(str, LambdaUri.get_function_name(authorizer_uri))
        identity_source_list = []
        if identity_source_template:
            for identity_source in identity_source_template.split(','):
                trimmed_id_source = identity_source.strip()
                if not IdentitySourceValidator.validate_identity_source(trimmed_id_source):
                    raise InvalidSamTemplateException(f'Lambda Authorizer {logical_id} does not contain valid identity sources.', Route.API)
                identity_source_list.append(trimmed_id_source)
        validation_expression = properties.get(LambdaAuthorizerV1Validator.AUTHORIZER_VALIDATION)
        lambda_authorizer = LambdaAuthorizer(payload_version='1.0', authorizer_name=name, type=authorizer_type, lambda_name=function_name, identity_sources=identity_source_list, validation_string=validation_expression)
        collector.add_authorizers(rest_api_id, {logical_id: lambda_authorizer})

    @staticmethod
    def _extract_cfn_gateway_v2_authorizer(logical_id: str, resource: dict, collector: ApiCollector) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Extract Authorizers from AWS::ApiGatewayV2::Authorizer and add them to the collector.\n\n        Parameters\n        ----------\n        logical_id: str\n            The logical ID of the Authorizer\n        resource: dict\n            The attributes for the Authorizer\n        collector: ApiCollector\n            ApiCollector to save Authorizers into\n        '
        if not LambdaAuthorizerV2Validator.validate(logical_id, resource):
            return
        properties = resource.get('Properties', {})
        authorizer_type = properties.get(LambdaAuthorizerV2Validator.AUTHORIZER_V2_TYPE, '').lower()
        api_id = properties.get(LambdaAuthorizerV2Validator.AUTHORIZER_V2_API)
        name = properties.get(LambdaAuthorizerV2Validator.AUTHORIZER_NAME)
        authorizer_uri = properties.get(LambdaAuthorizerV2Validator.AUTHORIZER_AUTHORIZER_URI)
        identity_sources = properties.get(LambdaAuthorizerV2Validator.AUTHORIZER_IDENTITY_SOURCE, [])
        payload_version = properties.get(LambdaAuthorizerV2Validator.AUTHORIZER_V2_PAYLOAD, LambdaAuthorizer.PAYLOAD_V2)
        simple_responses = properties.get(LambdaAuthorizerV2Validator.AUTHORIZER_V2_SIMPLE_RESPONSE, False)
        function_name = cast(str, LambdaUri.get_function_name(authorizer_uri))
        lambda_authorizer = LambdaAuthorizer(payload_version=payload_version, authorizer_name=name, type=authorizer_type, lambda_name=function_name, identity_sources=identity_sources, use_simple_response=simple_responses)
        collector.add_authorizers(api_id, {logical_id: lambda_authorizer})

    @staticmethod
    def _extract_cloud_formation_route(stack_path: str, logical_id: str, api_resource: Dict[str, Any], collector: ApiCollector, cwd: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Extract APIs from AWS::ApiGateway::RestApi resource by reading and parsing Swagger documents. The result is\n        added to the collector.\n\n        Parameters\n        ----------\n        stack_path : str\n            Path of the stack the resource is located\n\n        logical_id : str\n            Logical ID of the resource\n        api_resource : dict\n            Resource definition, including its properties\n        collector : ApiCollector\n            Instance of the API collector that where we will save the API information\n        cwd : Optional[str]\n            An optional string to override the current working directory\n        '
        properties = api_resource.get('Properties', {})
        body = properties.get('Body')
        body_s3_location = properties.get('BodyS3Location')
        binary_media = properties.get('BinaryMediaTypes', [])
        if not body and (not body_s3_location):
            LOG.debug("Skipping resource '%s'. Swagger document not found in Body and BodyS3Location", logical_id)
            return
        CfnBaseApiProvider.extract_swagger_route(stack_path, logical_id, body, body_s3_location, binary_media, collector, cwd)

    @staticmethod
    def _extract_cloud_formation_stage(resources: Dict[str, Dict], stage_resource: Dict, collector: ApiCollector) -> None:
        if False:
            return 10
        '\n         Extract the stage from AWS::ApiGateway::Stage resource by reading and adds it to the collector.\n         Parameters\n        ----------\n         resources: dict\n             All Resource definition, including its properties\n\n         stage_resource : dict\n             Stage Resource definition, including its properties\n\n         collector : ApiCollector\n             Instance of the API collector that where we will save the API information\n        '
        properties = stage_resource.get('Properties', {})
        stage_name = properties.get('StageName')
        stage_variables = properties.get('Variables')
        logical_id = properties.get('RestApiId')
        if not logical_id:
            raise InvalidSamTemplateException('The AWS::ApiGateway::Stage must have a RestApiId property')
        rest_api_resource_type = resources.get(logical_id, {}).get('Type')
        if rest_api_resource_type != AWS_APIGATEWAY_RESTAPI:
            raise InvalidSamTemplateException('The AWS::ApiGateway::Stage must have a valid RestApiId that points to RestApi resource {}'.format(logical_id))
        collector.stage_name = stage_name
        collector.stage_variables = stage_variables

    def _extract_cloud_formation_method(self, stack_path: str, resources: Dict[str, Dict], logical_id: str, method_resource: Dict, collector: ApiCollector) -> None:
        if False:
            return 10
        '\n        Extract APIs from AWS::ApiGateway::Method and work backwards up the tree to resolve and find the true path.\n\n        Parameters\n        ----------\n        stack_path : str\n            Path of the stack the resource is located\n\n        resources: dict\n            All Resource definition, including its properties\n\n        logical_id : str\n            Logical ID of the resource\n\n        method_resource : dict\n            Resource definition, including its properties\n\n        collector : ApiCollector\n            Instance of the API collector that where we will save the API information\n        '
        properties = method_resource.get('Properties', {})
        resource_id = properties.get('ResourceId')
        rest_api_id = properties.get('RestApiId')
        method = properties.get('HttpMethod')
        operation_name = properties.get('OperationName')
        resource_path = '/'
        if isinstance(resource_id, str):
            resource = resources.get(resource_id)
            if resource:
                resource_path = self.resolve_resource_path(resources, resource, '')
            else:
                resource_path = resource_id
        integration = properties.get('Integration', {})
        content_type = integration.get('ContentType')
        cors = None
        integration_responses = integration.get('IntegrationResponses')
        if integration_responses:
            for responses in integration_responses:
                response_parameters = responses.get('ResponseParameters')
                if response_parameters:
                    cors = self.extract_cors_from_method(response_parameters)
        if cors:
            collector.cors = cors
        content_handling = integration.get('ContentHandling')
        if content_handling == CfnApiProvider.METHOD_BINARY_TYPE and content_type:
            collector.add_binary_media_types(logical_id, [content_type])
        authorizer_name = properties.get(CfnApiProvider._METHOD_AUTHORIZER_ID)
        routes = Route(methods=[method], function_name=self._get_integration_function_name(integration), path=resource_path, operation_name=operation_name, stack_path=stack_path, authorizer_name=authorizer_name)
        collector.add_routes(rest_api_id, [routes])

    def _extract_cfn_gateway_v2_api(self, stack_path: str, logical_id: str, api_resource: Dict, collector: ApiCollector, cwd: Optional[str]=None) -> None:
        if False:
            while True:
                i = 10
        '\n        Extract APIs from AWS::ApiGatewayV2::Api resource by reading and parsing Swagger documents. The result is\n        added to the collector. If the Swagger documents is not available, it can add a catch-all route based on\n        the target function.\n\n        Parameters\n        ----------\n        stack_path : str\n            Path of the stack the resource is located\n\n        logical_id : str\n            Logical ID of the resource\n        api_resource : dict\n            Resource definition, including its properties\n        collector : ApiCollector\n            Instance of the API collector that where we will save the API information\n        cwd : Optional[str]\n            An optional string to override the current working directory\n        '
        properties = api_resource.get('Properties', {})
        body = properties.get('Body')
        body_s3_location = properties.get('BodyS3Location')
        cors = self.extract_cors_http(properties.get('CorsConfiguration'))
        target = properties.get('Target')
        route_key = properties.get('RouteKey')
        protocol_type = properties.get('ProtocolType')
        if not body and (not body_s3_location):
            LOG.debug("Swagger document not found in Body and BodyS3Location for resource '%s'.", logical_id)
            if cors:
                collector.cors = cors
            if target and protocol_type == CfnApiProvider.HTTP_API_PROTOCOL_TYPE:
                (method, path) = self._parse_route_key(route_key)
                routes = Route(methods=[method], path=path, function_name=LambdaUri.get_function_name(target), event_type=Route.HTTP, stack_path=stack_path)
                collector.add_routes(logical_id, [routes])
            return
        CfnBaseApiProvider.extract_swagger_route(stack_path, logical_id, body, body_s3_location, None, collector, cwd, Route.HTTP)

    def _extract_cfn_gateway_v2_route(self, stack_path: str, resources: Dict[str, Dict], logical_id: str, route_resource: Dict, collector: ApiCollector) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Extract APIs from AWS::ApiGatewayV2::Route, and link it with the integration resource to get the lambda\n        function.\n\n        Parameters\n        ----------\n        stack_path : str\n            Path of the stack the resource is located\n\n        resources: dict\n            All Resource definition, including its properties\n\n        logical_id : str\n            Logical ID of the resource\n\n        route_resource : dict\n            Resource definition, including its properties\n\n        collector : ApiCollector\n            Instance of the API collector that where we will save the API information\n        '
        properties = route_resource.get('Properties', {})
        api_id = properties.get('ApiId')
        route_key = properties.get('RouteKey')
        integration_target = properties.get('Target')
        operation_name = properties.get('OperationName')
        if integration_target:
            (function_name, payload_format_version) = self._get_route_function_name(resources, integration_target)
        else:
            LOG.debug("Skipping The AWS::ApiGatewayV2::Route '%s', as it does not contain an integration for a Lambda Function", logical_id)
            return
        (method, path) = self._parse_route_key(route_key)
        if not route_key or not method or (not path):
            LOG.debug("The AWS::ApiGatewayV2::Route '%s' does not have a correct route key '%s'", logical_id, route_key)
            raise InvalidSamTemplateException('The AWS::ApiGatewayV2::Route {} does not have a correct route key {}'.format(logical_id, route_key))
        authorizer_name = properties.get(CfnApiProvider._ROUTE_AUTHORIZER_ID)
        routes = Route(methods=[method], path=path, function_name=function_name, event_type=Route.HTTP, payload_format_version=payload_format_version, operation_name=operation_name, stack_path=stack_path, authorizer_name=authorizer_name)
        collector.add_routes(api_id, [routes])

    def resolve_resource_path(self, resources: Dict[str, Dict], resource: Dict, current_path: str) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Extract path from the Resource object by going up the tree\n\n        Parameters\n        ----------\n        resources: dict\n            Dictionary containing all the resources to resolve\n\n        resource : dict\n            AWS::ApiGateway::Resource definition and its properties\n\n        current_path : str\n            Current path resolved so far\n        '
        properties = resource.get('Properties', {})
        parent_id = cast(str, properties.get('ParentId'))
        resource_path = cast(str, properties.get('PathPart'))
        parent = resources.get(parent_id)
        if parent:
            return self.resolve_resource_path(resources, parent, '/' + resource_path + current_path)
        if parent_id:
            return parent_id + resource_path + current_path
        return '/' + resource_path + current_path

    @staticmethod
    def _extract_cfn_gateway_v2_stage(resources: Dict[str, Dict], stage_resource: Dict, collector: ApiCollector) -> None:
        if False:
            while True:
                i = 10
        '\n         Extract the stage from AWS::ApiGatewayV2::Stage resource by reading and adds it to the collector.\n         Parameters\n        ----------\n         resources: dict\n             All Resource definition, including its properties\n\n         stage_resource : dict\n             Stage Resource definition, including its properties\n\n         collector : ApiCollector\n             Instance of the API collector that where we will save the API information\n        '
        properties = stage_resource.get('Properties', {})
        stage_name = properties.get('StageName')
        stage_variables = properties.get('Variables')
        api_id = properties.get('ApiId')
        if not api_id:
            raise InvalidSamTemplateException('The AWS::ApiGatewayV2::Stage must have a ApiId property')
        api_resource_type = resources.get(api_id, {}).get('Type')
        if api_resource_type != AWS_APIGATEWAY_V2_API:
            raise InvalidSamTemplateException('The AWS::ApiGatewayV2::Stage must have a valid ApiId that points to Api resource {}'.format(api_id))
        collector.stage_name = stage_name
        collector.stage_variables = stage_variables

    @staticmethod
    def _get_integration_function_name(integration: Dict) -> Optional[str]:
        if False:
            while True:
                i = 10
        '\n        Tries to parse the Lambda Function name from the Integration defined in the method configuration. Integration\n        configuration. We care only about Lambda integrations, which are of type aws_proxy, and ignore the rest.\n        Integration URI is complex and hard to parse. Hence we do our best to extract function name out of\n        integration URI. If not possible, we return None.\n\n        Parameters\n        ----------\n        integration : Dict\n            the integration defined in the method configuration\n\n        Returns\n        -------\n        string or None\n            Lambda function name, if possible. None, if not.\n        '
        if integration and isinstance(integration, dict):
            uri: str = cast(str, integration.get('Uri'))
            return LambdaUri.get_function_name(uri)
        return None

    @staticmethod
    def _get_route_function_name(resources: Dict[str, Dict], integration_target: str) -> Tuple[Optional[str], Optional[str]]:
        if False:
            while True:
                i = 10
        '\n        Look for the APIGateway integration resource based on the input integration_target, then try to parse the\n        lambda function from the the integration resource properties.\n        It also gets the Payload format version from the API Gateway integration resource.\n\n        Parameters\n        ----------\n        resources : dict\n            dictionary of all resources.\n\n        integration_target : str\n            the path of the HTTP Gateway integration resource\n\n        Returns\n        -------\n        string or None, string or None\n            Lambda function name, if possible. None, if not.\n            Payload format version, if possible. None, if not\n        '
        integration_id = integration_target.split('/')[1].strip()
        integration_resource = resources.get(integration_id, {})
        resource_type = integration_resource.get('Type')
        if resource_type == AWS_APIGATEWAY_V2_INTEGRATION:
            properties = integration_resource.get('Properties', {})
            integration_uri = properties.get('IntegrationUri')
            payload_format_version = properties.get('PayloadFormatVersion')
            if integration_uri and isinstance(integration_uri, str):
                return (LambdaUri.get_function_name(integration_uri), payload_format_version)
        return (None, None)

    @staticmethod
    def _parse_route_key(route_key: Optional[str]) -> Tuple[str, str]:
        if False:
            print('Hello World!')
        '\n        parse the route key, and return the methods && path.\n        route key should be in format "Http_method Path" or to equal "$default"\n        if the route key is $default, return \'X-AMAZON-APIGATEWAY-ANY-METHOD\' as a method && $default as a path\n        else we will split the route key on space and use the specified method and path\n\n        Parameters\n        ----------\n        route_key : str\n            the defined route key.\n\n        Returns\n        -------\n        string, string\n            method as defined in the route key or X-AMAZON-APIGATEWAY-ANY-METHOD .\n            route key path if defined or $default.\n        '
        if not route_key or route_key == '$default':
            return ('X-AMAZON-APIGATEWAY-ANY-METHOD', '$default')
        [method, path] = route_key.split()
        return (method, path)