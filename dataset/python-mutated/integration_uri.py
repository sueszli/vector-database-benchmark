"""
Handles parsing of Swagger Integration data. This contains the arn of the Lambda function it connects to,
integration type, response template etc.
"""
import logging
import re
from enum import Enum
from typing import Optional
LOG = logging.getLogger(__name__)

class LambdaUri:
    """
    Purely static class that helps you parse Lambda Function Integration URI ARN
    """
    _FN_SUB = 'Fn::Sub'
    _REGEX_GET_FUNCTION_ARN = 'arn:aws:lambda:[^:]+:[^:]+:function:[^:/]+(?::[^/]+)?'
    _REGEX_GET_FUNCTION_ARN_GENERIC = '/functions/([^/]+)/invocations'
    _REGEX_GET_FUNCTION_NAME = '.*:function:([^:]*)'
    _REGEX_STAGE_VARIABLE = '\\$\\{stageVariables\\..+\\}'
    _REGEX_VALID_FUNCTION_NAME = '([a-zA-Z0-9-_]+)'
    _REGEX_SUB_FUNCTION_ARN = '\\$\\{([A-Za-z0-9]+)\\.(Arn|Alias)\\}'

    @staticmethod
    def get_function_name(integration_uri) -> Optional[str]:
        if False:
            while True:
                i = 10
        '\n        Gets the name of the function from the Integration URI ARN. This is a best effort service which returns None\n        if function name could not be parsed. This can happen when the ARN is an intrinsic function which is too\n        complex or the ARN is not a Lambda integration.\n\n        Parameters\n        ----------\n        integration_uri : basestring or dict\n            Integration URI data extracted from Swagger dictionary. This could be a string of the ARN or an intrinsic\n            function that will resolve to the ARN\n\n        Returns\n        -------\n        basestring or None\n            If the function name could be parsed out of the Integration URI ARN. None, otherwise\n        '
        arn = LambdaUri._get_function_arn(integration_uri)
        LOG.debug('Extracted Function ARN: %s', arn)
        return LambdaUri._get_function_name_from_arn(arn)

    @staticmethod
    def _get_function_arn(uri_data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Integration URI can be expressed in various shapes and forms. This method normalizes the Integration URI ARN\n        and returns the Lambda Function ARN. Here are the different forms of Integration URI ARN:\n\n        - String:\n            - Fully resolved ARN\n            - ARN with Stage Variables:\n              Ex: arn:aws:apigateway:ap-southeast-2:lambda:path/2015-03-31/functions/arn:aws:lambda:ap-southeast-2:123456789012:function:${stageVariables.PostFunctionName}/invocations  # pylint: disable=line-too-long # noqa\n\n        - Dictionary: Usually contains intrinsic functions\n\n            - Fn::Sub:\n              Example:\n              {\n                "Fn::Sub":\n                  "arn:aws:apigateway:${AWS::Region}:lambda:path/2015-03-31/functions/${LambdaFunction.Arn}/invocations"\n              }\n\n            - Fn::Join: **Unsupported**. It is very hard to combine the joins into one string especially when\n              certain properties are resolved only at runtime.\n\n            - Ref, Fn::GetAtt: **Unsupported**. Impossible to use these intrinsics with integration URI. CFN doesn\'t\n              support this functionality.\n\n        Note\n        ~~~~\n        This method supports only a very restricted subset of intrinsic functions with Swagger document. This is the\n        best we can do without implementing a full blown intrinsic function resolution module.\n\n        Parameters\n        ----------\n        uri_data : string or dict\n            Value of Integration URI. It can either be a string or an intrinsic function that resolves to a string\n\n        Returns\n        -------\n        basestring or None\n            Lambda Function ARN extracted from Integration URI. None, if it cannot get function Arn\n        '
        if not uri_data:
            return None
        if LambdaUri._is_sub_intrinsic(uri_data):
            uri_data = LambdaUri._resolve_fn_sub(uri_data)
            LOG.debug('Resolved Sub intrinsic function: %s', uri_data)
        if not isinstance(uri_data, str):
            LOG.debug('This Integration URI format is not supported: %s', uri_data)
            return None
        matches = re.search(LambdaUri._REGEX_GET_FUNCTION_ARN, uri_data)
        if matches:
            return matches.group(0)
        matches = re.search(LambdaUri._REGEX_GET_FUNCTION_ARN_GENERIC, uri_data)
        if matches:
            return matches.group(1)
        LOG.debug('Ignoring Integration URI because it is not a Lambda Function integration: %s', uri_data)
        return None

    @staticmethod
    def _get_function_name_from_arn(function_arn) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        '\n        Given the integration ARN, extract the Lambda function name from the ARN. If there\n        are stage variables, or other unsupported formats, this function will return None.\n\n        Parameters\n        ----------\n        function_arn : basestring or None\n            Function ARN from the swagger document\n\n        Returns\n        -------\n        basestring or None\n            Function name of this integration. None if the ARN is not parsable\n        '
        if not function_arn:
            return None
        matches = re.match(LambdaUri._REGEX_GET_FUNCTION_NAME, function_arn)
        if not matches or not matches.groups():
            LOG.debug('No Lambda function ARN defined for integration containing ARN %s', function_arn)
            return None
        groups = matches.groups()
        maybe_function_name: str = groups[0]
        if re.match(LambdaUri._REGEX_STAGE_VARIABLE, maybe_function_name):
            LOG.debug('Stage variables are not supported. Ignoring integration with function ARN %s', function_arn)
            return None
        if re.match(LambdaUri._REGEX_VALID_FUNCTION_NAME, maybe_function_name):
            return maybe_function_name
        LOG.debug('Ignoring integration ARN. Unable to parse Function Name from function arn %s', function_arn)
        return None

    @staticmethod
    def _resolve_fn_sub(uri_data):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tries to resolve an Integration URI which contains Fn::Sub intrinsic function. This method tries to resolve\n        and produce a string output.\n\n        Example:\n        {\n          "Fn::Sub":\n            "arn:aws:apigateway:${AWS::Region}:lambda:path/2015-03-31/functions/${LambdaFunction.Arn}/invocations"\n        }\n\n        Fn::Sub Processing:\n        ~~~~~~~~~~~~~~~~~~\n\n        If this is a Fn::Sub, resolve as following:\n            1. Get the ARN String:\n                - If Sub is using the array syntax, then use element which is a string.\n                - If Sub is using string syntax, then just use the string.\n            2. If there is a ${XXX.Arn} then replace it with a dummy ARN\n            3. Otherwise skip it\n\n        .. code:\n            Input:\n            {\n              "Fn::Sub":\n                "arn:aws:apigateway:${AWS::Region}:lambda:path/2015-03-31/functions/${LambdaFunction.Arn}/invocations"\n            }\n\n            Output: "arn:aws:apigateway:${AWS::Region}:lambda:path/2015-03-31/functions/arn:aws:lambda:us-east-1:LambdaFunction/invocations"  # pylint: disable=line-too-long # noqa\n\n        Note\n        ~~~~\n        This method supports only a very restricted subset of intrinsic functions with Swagger document. This is the\n        best we can do without implementing a full blown intrinsic function resolution module.\n\n        Parameters\n        ----------\n        uri_data : string or dict\n            Value of Integration URI. It can either be a string or an intrinsic function that resolves to a string\n\n        Returns\n        -------\n        string\n            Integration URI as a string, if we were able to resolve the Sub intrinsic\n        dict\n            Input data is returned unmodified if we are unable to resolve the intrinsic\n        '
        arn = uri_data[LambdaUri._FN_SUB]
        if isinstance(arn, list):
            arn = arn[0]
        if not isinstance(arn, str):
            LOG.debug('Unable to resolve Fn::Sub value for integration URI: %s', uri_data)
            return uri_data
        lambda_function_arn_template = 'arn:aws:lambda:${AWS::Region}:123456789012:function:\\1'
        return re.sub(LambdaUri._REGEX_SUB_FUNCTION_ARN, lambda_function_arn_template, arn)

    @staticmethod
    def _is_sub_intrinsic(data):
        if False:
            while True:
                i = 10
        '\n        Is this input data a Fn::Sub intrinsic function\n\n        Parameters\n        ----------\n        data\n            Data to check\n\n        Returns\n        -------\n        bool\n            True if the data Fn::Sub intrinsic function\n        '
        return isinstance(data, dict) and len(data) == 1 and (LambdaUri._FN_SUB in data)

class IntegrationType(Enum):
    aws_proxy = 'aws_proxy'
    mock = 'mock'