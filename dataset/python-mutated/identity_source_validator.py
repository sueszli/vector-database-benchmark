"""
Handles the validation of identity sources
"""
import re
from samcli.local.apigw.route import Route

class IdentitySourceValidator:
    API_GATEWAY_V1_QUERY_REGEX = re.compile('method\\.request\\.querystring\\.[a-zA-Z0-9._-]+$')
    API_GATEWAY_V1_HEADER_REGEX = re.compile('method\\.request\\.header\\.[a-zA-Z0-9._-]+$')
    API_GATEWAY_V1_CONTEXT_REGEX = re.compile('context\\.[a-zA-Z0-9._-]+$')
    API_GATEWAY_V1_STAGE_REGEX = re.compile('stageVariables\\.[a-zA-Z0-9._-]+$')
    API_GATEWAY_V2_QUERY_REGEX = re.compile('\\$request\\.querystring\\.[a-zA-Z0-9._-]+$')
    API_GATEWAY_V2_HEADER_REGEX = re.compile('\\$request\\.header\\.[a-zA-Z0-9._-]+$')
    API_GATEWAY_V2_CONTEXT_REGEX = re.compile('\\$context\\.[a-zA-Z0-9._-]+$')
    API_GATEWAY_V2_STAGE_REGEX = re.compile('\\$stageVariables\\.[a-zA-Z0-9._-]+$')
    API_GATEWAY_VALIDATION_LIST = {Route.API: [API_GATEWAY_V1_QUERY_REGEX, API_GATEWAY_V1_HEADER_REGEX, API_GATEWAY_V1_CONTEXT_REGEX, API_GATEWAY_V1_STAGE_REGEX], Route.HTTP: [API_GATEWAY_V2_QUERY_REGEX, API_GATEWAY_V2_HEADER_REGEX, API_GATEWAY_V2_CONTEXT_REGEX, API_GATEWAY_V2_STAGE_REGEX]}

    @staticmethod
    def validate_identity_source(identity_source: str, event_type: str=Route.API) -> bool:
        if False:
            while True:
                i = 10
        '\n        Validates if the identity source is valid for the provided event type\n\n        Parameters\n        ----------\n        identity_source: str\n            The identity source to validate\n        event_type: str\n            The type of API Gateway to validate against (API or HTTP)\n\n        Returns\n        -------\n        bool\n            True if the identity source is valid\n        '
        for regex in IdentitySourceValidator.API_GATEWAY_VALIDATION_LIST[event_type]:
            if regex.match(identity_source):
                return True
        return False