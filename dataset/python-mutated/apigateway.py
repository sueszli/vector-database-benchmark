from . import integer_range, json_checker, positive_integer, tags_or_list

def dict_or_string(x):
    if False:
        print('Hello World!')
    '\n    Property: Model.Schema\n    '
    if isinstance(x, (dict, str)):
        return x
    raise TypeError(f'Value {x} of type {type(x)} must be either dict or str')

def validate_tags_or_list(x):
    if False:
        return 10
    '\n    Property: StageDescription.Tags\n    Property: Stage.Tags\n    '
    return tags_or_list(x)

def validate_timeout_in_millis(x):
    if False:
        while True:
            i = 10
    '\n    Property: Integration.TimeoutInMillis\n    '
    return integer_range(50, 29000)(x)

def validate_authorizer_ttl(ttl_value):
    if False:
        print('Hello World!')
    'Validate authorizer ttl timeout\n    :param ttl_value: The TTL timeout in seconds\n    :return: The provided TTL value if valid\n    Property: Authorizer.AuthorizerResultTtlInSeconds\n    '
    ttl_value = int(positive_integer(ttl_value))
    if ttl_value > 3600:
        raise ValueError('The AuthorizerResultTtlInSeconds should be <= 3600')
    return ttl_value

def validate_gateway_response_type(response_type):
    if False:
        i = 10
        return i + 15
    'Validate response type\n    :param response_type: The GatewayResponse response type\n    :return: The provided value if valid\n    Property: GatewayResponse.ResponseType\n    '
    valid_response_types = ['ACCESS_DENIED', 'API_CONFIGURATION_ERROR', 'AUTHORIZER_FAILURE', 'AUTHORIZER_CONFIGURATION_ERROR', 'BAD_REQUEST_PARAMETERS', 'BAD_REQUEST_BODY', 'DEFAULT_4XX', 'DEFAULT_5XX', 'EXPIRED_TOKEN', 'INVALID_SIGNATURE', 'INTEGRATION_FAILURE', 'INTEGRATION_TIMEOUT', 'INVALID_API_KEY', 'MISSING_AUTHENTICATION_TOKEN', 'QUOTA_EXCEEDED', 'REQUEST_TOO_LARGE', 'RESOURCE_NOT_FOUND', 'THROTTLED', 'UNAUTHORIZED', 'UNSUPPORTED_MEDIA_TYPE']
    if response_type not in valid_response_types:
        raise ValueError('{} is not a valid ResponseType'.format(response_type))
    return response_type

def validate_model(self):
    if False:
        return 10
    '\n    Class: Model\n    '
    name = 'Schema'
    if name in self.properties:
        schema = self.properties.get(name)
        self.properties[name] = json_checker(schema)