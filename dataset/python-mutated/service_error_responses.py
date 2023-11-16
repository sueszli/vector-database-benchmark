"""Class container to hold common Service Responses"""
from flask import Response, jsonify, make_response

class ServiceErrorResponses:
    _NO_LAMBDA_INTEGRATION = {'message': 'No function defined for resource method'}
    _MISSING_AUTHENTICATION = {'message': 'Missing Authentication Token'}
    _LAMBDA_FAILURE = {'message': 'Internal server error'}
    _MISSING_LAMBDA_AUTH_IDENTITY_SOURCES = {'message': 'Unauthorized'}
    _LAMBDA_AUTHORIZER_NOT_AUTHORIZED = {'message': 'User is not authorized to access this resource'}
    HTTP_STATUS_CODE_500 = 500
    HTTP_STATUS_CODE_501 = 501
    HTTP_STATUS_CODE_502 = 502
    HTTP_STATUS_CODE_403 = 403
    HTTP_STATUS_CODE_401 = 401

    @staticmethod
    def lambda_authorizer_unauthorized() -> Response:
        if False:
            return 10
        '\n        Constructs a Flask response for when a route invokes a Lambda Authorizer, but\n        is the identity sources provided are not authorized for that method\n\n        Returns\n        -------\n        Response\n            A Flask Response object\n        '
        response_data = jsonify(ServiceErrorResponses._LAMBDA_AUTHORIZER_NOT_AUTHORIZED)
        return make_response(response_data, ServiceErrorResponses.HTTP_STATUS_CODE_403)

    @staticmethod
    def missing_lambda_auth_identity_sources() -> Response:
        if False:
            return 10
        '\n        Constructs a Flask response for when a route contains a Lambda Authorizer\n        but is missing the required identity services\n\n        Returns\n        -------\n        Response\n            A Flask Response object\n        '
        response_data = jsonify(ServiceErrorResponses._MISSING_LAMBDA_AUTH_IDENTITY_SOURCES)
        return make_response(response_data, ServiceErrorResponses.HTTP_STATUS_CODE_401)

    @staticmethod
    def lambda_failure_response(*args):
        if False:
            while True:
                i = 10
        '\n        Helper function to create a Lambda Failure Response\n\n        :return: A Flask Response\n        '
        response_data = jsonify(ServiceErrorResponses._LAMBDA_FAILURE)
        return make_response(response_data, ServiceErrorResponses.HTTP_STATUS_CODE_502)

    @staticmethod
    def lambda_body_failure_response(*args):
        if False:
            print('Hello World!')
        '\n        Helper function to create a Lambda Body Failure Response\n\n        :return: A Flask Response\n        '
        response_data = jsonify(ServiceErrorResponses._LAMBDA_FAILURE)
        return make_response(response_data, ServiceErrorResponses.HTTP_STATUS_CODE_500)

    @staticmethod
    def not_implemented_locally(message):
        if False:
            i = 10
            return i + 15
        '\n        Constructs a Flask Response for for when a Lambda function functionality is\n        not implemented\n\n        :return: a Flask Response\n        '
        exception_dict = {'message': message}
        response_data = jsonify(exception_dict)
        return make_response(response_data, ServiceErrorResponses.HTTP_STATUS_CODE_501)

    @staticmethod
    def lambda_not_found_response(*args):
        if False:
            print('Hello World!')
        '\n        Constructs a Flask Response for when a Lambda function is not found for an endpoint\n\n        :return: a Flask Response\n        '
        response_data = jsonify(ServiceErrorResponses._NO_LAMBDA_INTEGRATION)
        return make_response(response_data, ServiceErrorResponses.HTTP_STATUS_CODE_502)

    @staticmethod
    def route_not_found(*args):
        if False:
            for i in range(10):
                print('nop')
        '\n        Constructs a Flask Response for when a API Route (path+method) is not found. This is usually\n        HTTP 404 but with API Gateway this is a HTTP 403 (https://forums.aws.amazon.com/thread.jspa?threadID=2166840)\n\n        :return: a Flask Response\n        '
        response_data = jsonify(ServiceErrorResponses._MISSING_AUTHENTICATION)
        return make_response(response_data, ServiceErrorResponses.HTTP_STATUS_CODE_403)