from . import integer_range, json_checker, positive_integer

def dict_or_string(x):
    if False:
        print('Hello World!')
    '\n    Property: Model.Schema\n    '
    if isinstance(x, (dict, str)):
        return x
    raise TypeError(f'Value {x} of type {type(x)} must be either dict or str')

def validate_timeout_in_millis(x):
    if False:
        i = 10
        return i + 15
    '\n    Property: Integration.TimeoutInMillis\n    '
    return integer_range(50, 29000)(x)

def validate_integration_type(integration_type):
    if False:
        for i in range(10):
            print('nop')
    '\n    Property: Integration.IntegrationType\n    '
    valid_integration_types = ['AWS', 'AWS_PROXY', 'HTTP', 'HTTP_PROXY', 'MOCK']
    if integration_type not in valid_integration_types:
        raise ValueError('{} is not a valid IntegrationType'.format(integration_type))
    return integration_type

def validate_authorizer_type(authorizer_type):
    if False:
        i = 10
        return i + 15
    '\n    Property: Authorizer.AuthorizerType\n    '
    valid_authorizer_types = ['REQUEST', 'JWT']
    if authorizer_type not in valid_authorizer_types:
        raise ValueError('{} is not a valid AuthorizerType'.format(authorizer_type))
    return authorizer_type

def validate_logging_level(logging_level):
    if False:
        i = 10
        return i + 15
    '\n    Property: RouteSettings.LoggingLevel\n    '
    valid_logging_levels = ['INFO', 'ERROR', 'OFF']
    if logging_level not in valid_logging_levels:
        raise ValueError('{} is not a valid LoggingLevel'.format(logging_level))
    return logging_level

def validate_passthrough_behavior(passthrough_behavior):
    if False:
        for i in range(10):
            print('nop')
    '\n    Property: Integration.PassthroughBehavior\n    '
    valid_passthrough_behaviors = ['WHEN_NO_MATCH', 'WHEN_NO_TEMPLATES', 'NEVER']
    if passthrough_behavior not in valid_passthrough_behaviors:
        raise ValueError('{} is not a valid PassthroughBehavior'.format(passthrough_behavior))
    return passthrough_behavior

def validate_content_handling_strategy(content_handling_strategy):
    if False:
        print('Hello World!')
    '\n    Property: Integration.ContentHandlingStrategy\n    Property: IntegrationResponse.ContentHandlingStrategy\n    '
    valid_handling_strategy_values = ['CONVERT_TO_TEXT', 'CONVERT_TO_BINARY']
    if content_handling_strategy not in valid_handling_strategy_values:
        raise ValueError('{} is not a valid ContentHandlingStrategy'.format(content_handling_strategy))
    return content_handling_strategy

def validate_authorizer_ttl(ttl_value):
    if False:
        print('Hello World!')
    'Validate authorizer ttl timeout\n    :param ttl_value: The TTL timeout in seconds\n    :return: The provided TTL value if valid\n    Property: Authorizer.AuthorizerResultTtlInSeconds\n    '
    ttl_value = int(positive_integer(ttl_value))
    if ttl_value > 3600:
        raise ValueError('The AuthorizerResultTtlInSeconds should be <= 3600')
    return ttl_value

def validate_model(self):
    if False:
        i = 10
        return i + 15
    '\n    Class: Model\n    '
    name = 'Schema'
    if name in self.properties:
        schema = self.properties.get(name)
        self.properties[name] = json_checker(schema)