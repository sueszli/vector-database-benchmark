import base64
import json
from json import JSONDecodeError
from metaflow.exception import MetaflowException
from metaflow.metaflow_config import AWS_SECRETS_MANAGER_DEFAULT_REGION
from metaflow.plugins.secrets import SecretsProvider
import re

class MetaflowAWSSecretsManagerBadResponse(MetaflowException):
    """Raised when the response from AWS Secrets Manager is not valid in some way"""

class MetaflowAWSSecretsManagerDuplicateKey(MetaflowException):
    """Raised when the response from AWS Secrets Manager contains duplicate keys"""

class MetaflowAWSSecretsManagerJSONParseError(MetaflowException):
    """Raised when the SecretString response from AWS Secrets Manager is not valid JSON"""

class MetaflowAWSSecretsManagerNotJSONObject(MetaflowException):
    """Raised when the SecretString response from AWS Secrets Manager is not valid JSON object (dictionary)"""

def _sanitize_key_as_env_var(key):
    if False:
        for i in range(10):
            print('nop')
    "\n    Sanitize a key as an environment variable name.\n    This is purely a convenience trade-off to cover common cases well, vs. introducing\n    ambiguities (e.g. did the final '_' come from '.', or '-' or is original?).\n\n    1/27/2023(jackie):\n\n    We start with few rules and should *sparingly* add more over time.\n    Also, it's TBD whether all possible providers will share the same sanitization logic.\n    Therefore we will keep this function private for now\n    "
    return key.replace('-', '_').replace('.', '_').replace('/', '_')

class AwsSecretsManagerSecretsProvider(SecretsProvider):
    TYPE = 'aws-secrets-manager'

    def get_secret_as_dict(self, secret_id, options={}, role=None):
        if False:
            print('Hello World!')
        '\n        Reads a secret from AWS Secrets Manager and returns it as a dictionary of environment variables.\n\n        The secret payload from AWS is EITHER a string OR a binary blob.\n\n        If the secret contains a string payload ("SecretString"):\n        - if the `parse_secret_string_as_json` option is True (default):\n            {SecretString} will be parsed as a JSON. If successfully parsed, AND the JSON contains a\n            top-level object, each entry K/V in the object will also be converted to an entry in the result. V will\n            always be casted to a string (if not already a string).\n        - If `parse_secret_string_as_json` option is False:\n            {SecretString} will be returned as a single entry in the result, with the key being the secret_id.\n\n        Otherwise, the secret contains a binary blob payload ("SecretBinary"). In this case\n        - The result dic contains \'{SecretName}\': \'{SecretBinary}\', where {SecretBinary} is a base64-encoded string\n\n        All keys in the result are sanitized to be more valid environment variable names. This is done on a best effort\n        basis. Further validation is expected to be done by the invoking @secrets decorator itself.\n\n        :param secret_id: ARN or friendly name of the secret\n        :param options: unused\n        :param role: AWS IAM Role ARN to assume before reading the secret\n        :return: dict of environment variables. All keys and values are strings.\n        '
        import botocore
        from metaflow.plugins.aws.aws_client import get_aws_client
        effective_aws_region = None
        m = re.match('arn:aws:secretsmanager:([^:]+):', secret_id)
        if m:
            effective_aws_region = m.group(1)
        elif 'region' in options:
            effective_aws_region = options['region']
        else:
            effective_aws_region = AWS_SECRETS_MANAGER_DEFAULT_REGION
        try:
            secrets_manager_client = get_aws_client('secretsmanager', client_params={'region_name': effective_aws_region}, role_arn=role)
        except botocore.exceptions.NoRegionError:
            raise MetaflowException('Default region is not specified for AWS Secrets Manager. Please set METAFLOW_AWS_SECRETS_MANAGER_DEFAULT_REGION')
        result = {}

        def _sanitize_and_add_entry_to_result(k, v):
            if False:
                print('Hello World!')
            sanitized_k = _sanitize_key_as_env_var(k)
            if sanitized_k in result:
                raise MetaflowAWSSecretsManagerDuplicateKey("Duplicate key in secret: '%s' (sanitizes to '%s')" % (k, sanitized_k))
            result[sanitized_k] = v
        "\n        These are the exceptions that can be raised by the AWS SDK:\n        \n        SecretsManager.Client.exceptions.ResourceNotFoundException\n        SecretsManager.Client.exceptions.InvalidParameterException\n        SecretsManager.Client.exceptions.InvalidRequestException\n        SecretsManager.Client.exceptions.DecryptionFailure\n        SecretsManager.Client.exceptions.InternalServiceError\n        \n        Looks pretty informative already, so we won't catch here directly.\n        \n        1/27/2023(jackie) - We will evolve this over time as we learn more.\n        "
        response = secrets_manager_client.get_secret_value(SecretId=secret_id)
        if 'Name' not in response:
            raise MetaflowAWSSecretsManagerBadResponse("Secret 'Name' is missing in response")
        secret_name = response['Name']
        if 'SecretString' in response:
            secret_str = response['SecretString']
            if options.get('json', True):
                try:
                    obj = json.loads(secret_str)
                    if type(obj) == dict:
                        for (k, v) in obj.items():
                            _sanitize_and_add_entry_to_result(k, str(v))
                    else:
                        raise MetaflowAWSSecretsManagerNotJSONObject('Secret string is a JSON, but not an object (dict-like) - actual type %s.' % type(obj))
                except JSONDecodeError:
                    raise MetaflowAWSSecretsManagerJSONParseError('Secret string could not be parsed as JSON')
            else:
                if options.get('env_var_name'):
                    env_var_name = options['env_var_name']
                else:
                    env_var_name = secret_name
                _sanitize_and_add_entry_to_result(env_var_name, secret_str)
        elif 'SecretBinary' in response:
            if options.get('env_var_name'):
                env_var_name = options['env_var_name']
            else:
                env_var_name = secret_name
            _sanitize_and_add_entry_to_result(env_var_name, base64.b64encode(response['SecretBinary']).decode())
        else:
            raise MetaflowAWSSecretsManagerBadResponse("Secret response is missing both 'SecretString' and 'SecretBinary'")
        return result