"""Common Lambda Error Responses"""
import json
from collections import OrderedDict
from samcli.local.services.base_local_service import BaseLocalService

class LambdaErrorResponses:
    UnsupportedMediaTypeException = ('UnsupportedMediaType', 415)
    ServiceException = ('Service', 500)
    ResourceNotFoundException = ('ResourceNotFound', 404)
    InvalidRequestContentException = ('InvalidRequestContent', 400)
    NotImplementedException = ('NotImplemented', 501)
    PathNotFoundException = ('PathNotFoundLocally', 404)
    MethodNotAllowedException = ('MethodNotAllowedLocally', 405)
    USER_ERROR = 'User'
    SERVICE_ERROR = 'Service'
    LOCAL_SERVICE_ERROR = 'LocalService'
    CONTENT_TYPE = 'application/json'
    CONTENT_TYPE_HEADER_KEY = 'Content-Type'

    @staticmethod
    def resource_not_found(function_name):
        if False:
            while True:
                i = 10
        '\n        Creates a Lambda Service ResourceNotFound Response\n\n        Parameters\n        ----------\n        function_name str\n            Name of the function that was requested to invoke\n\n        Returns\n        -------\n        Flask.Response\n            A response object representing the ResourceNotFound Error\n        '
        exception_tuple = LambdaErrorResponses.ResourceNotFoundException
        return BaseLocalService.service_response(LambdaErrorResponses._construct_error_response_body(LambdaErrorResponses.USER_ERROR, 'Function not found: arn:aws:lambda:us-west-2:012345678901:function:{}'.format(function_name)), LambdaErrorResponses._construct_headers(exception_tuple[0]), exception_tuple[1])

    @staticmethod
    def invalid_request_content(message):
        if False:
            print('Hello World!')
        '\n        Creates a Lambda Service InvalidRequestContent Response\n\n        Parameters\n        ----------\n        message str\n            Message to be added to the body of the response\n\n        Returns\n        -------\n        Flask.Response\n            A response object representing the InvalidRequestContent Error\n        '
        exception_tuple = LambdaErrorResponses.InvalidRequestContentException
        return BaseLocalService.service_response(LambdaErrorResponses._construct_error_response_body(LambdaErrorResponses.USER_ERROR, message), LambdaErrorResponses._construct_headers(exception_tuple[0]), exception_tuple[1])

    @staticmethod
    def unsupported_media_type(content_type):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a Lambda Service UnsupportedMediaType Response\n\n        Parameters\n        ----------\n        content_type str\n            Content Type of the request that was made\n\n        Returns\n        -------\n        Flask.Response\n            A response object representing the UnsupportedMediaType Error\n        '
        exception_tuple = LambdaErrorResponses.UnsupportedMediaTypeException
        return BaseLocalService.service_response(LambdaErrorResponses._construct_error_response_body(LambdaErrorResponses.USER_ERROR, 'Unsupported content type: {}'.format(content_type)), LambdaErrorResponses._construct_headers(exception_tuple[0]), exception_tuple[1])

    @staticmethod
    def generic_service_exception(*args):
        if False:
            print('Hello World!')
        '\n        Creates a Lambda Service Generic ServiceException Response\n\n        Parameters\n        ----------\n        args list\n            List of arguments Flask passes to the method\n\n        Returns\n        -------\n        Flask.Response\n            A response object representing the GenericServiceException Error\n        '
        exception_tuple = LambdaErrorResponses.ServiceException
        return BaseLocalService.service_response(LambdaErrorResponses._construct_error_response_body(LambdaErrorResponses.SERVICE_ERROR, 'ServiceException'), LambdaErrorResponses._construct_headers(exception_tuple[0]), exception_tuple[1])

    @staticmethod
    def not_implemented_locally(message):
        if False:
            print('Hello World!')
        '\n        Creates a Lambda Service NotImplementedLocally Response\n\n        Parameters\n        ----------\n        message str\n            Message to be added to the body of the response\n\n        Returns\n        -------\n        Flask.Response\n            A response object representing the NotImplementedLocally Error\n        '
        exception_tuple = LambdaErrorResponses.NotImplementedException
        return BaseLocalService.service_response(LambdaErrorResponses._construct_error_response_body(LambdaErrorResponses.LOCAL_SERVICE_ERROR, message), LambdaErrorResponses._construct_headers(exception_tuple[0]), exception_tuple[1])

    @staticmethod
    def generic_path_not_found(*args):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a Lambda Service Generic PathNotFound Response\n\n        Parameters\n        ----------\n        args list\n            List of arguments Flask passes to the method\n\n        Returns\n        -------\n        Flask.Response\n            A response object representing the GenericPathNotFound Error\n        '
        exception_tuple = LambdaErrorResponses.PathNotFoundException
        return BaseLocalService.service_response(LambdaErrorResponses._construct_error_response_body(LambdaErrorResponses.LOCAL_SERVICE_ERROR, 'PathNotFoundException'), LambdaErrorResponses._construct_headers(exception_tuple[0]), exception_tuple[1])

    @staticmethod
    def generic_method_not_allowed(*args):
        if False:
            return 10
        '\n        Creates a Lambda Service Generic MethodNotAllowed Response\n\n        Parameters\n        ----------\n        args list\n            List of arguments Flask passes to the method\n\n        Returns\n        -------\n        Flask.Response\n            A response object representing the GenericMethodNotAllowed Error\n        '
        exception_tuple = LambdaErrorResponses.MethodNotAllowedException
        return BaseLocalService.service_response(LambdaErrorResponses._construct_error_response_body(LambdaErrorResponses.LOCAL_SERVICE_ERROR, 'MethodNotAllowedException'), LambdaErrorResponses._construct_headers(exception_tuple[0]), exception_tuple[1])

    @staticmethod
    def _construct_error_response_body(error_type, error_message):
        if False:
            while True:
                i = 10
        '\n        Constructs a string to be used in the body of the Response that conforms\n        to the structure of the Lambda Service Responses\n\n        Parameters\n        ----------\n        error_type str\n            The type of error\n        error_message str\n            Message of the error that occured\n\n        Returns\n        -------\n        str\n            str representing the response body\n        '
        return json.dumps(OrderedDict([('Type', error_type), ('Message', error_message)]))

    @staticmethod
    def _construct_headers(error_type):
        if False:
            print('Hello World!')
        "\n        Constructs Headers for the Local Lambda Error Response\n\n        Parameters\n        ----------\n        error_type str\n            Error type that occurred to be put into the 'x-amzn-errortype' header\n\n        Returns\n        -------\n        dict\n            Dict representing the Lambda Error Response Headers\n        "
        return {'x-amzn-errortype': error_type, 'Content-Type': 'application/json'}