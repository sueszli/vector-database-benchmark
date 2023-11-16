"""
Stub functions that are used by the AWS Secrets Manager unit tests.
"""
from test_tools.example_stubber import ExampleStubber

class SecretsManagerStubber(ExampleStubber):
    """
    A class that implements a variety of stub functions that are used by the
    AWS Secrets Manager unit tests.

    The stubbed functions all expect certain parameters to be passed to them as
    part of the tests, and will raise errors when the actual parameters differ from
    the expected.
    """

    def __init__(self, client, use_stubs=True):
        if False:
            while True:
                i = 10
        '\n        Initializes the object with a specific client and configures it for\n        stubbing or AWS passthrough.\n\n        :param client: A Boto3 Secrets Manager client.\n        :param use_stubs: When True, use stubs to intercept requests. Otherwise,\n                          pass requests through to AWS.\n        '
        super().__init__(client, use_stubs)

    def stub_create_secret(self, secret_name, secret_value, error_code=None):
        if False:
            print('Hello World!')
        expected_params = {'Name': secret_name}
        if isinstance(secret_value, str):
            expected_params['SecretString'] = secret_value
        elif isinstance(secret_value, bytes):
            expected_params['SecretBinary'] = secret_value
        response = {'Name': secret_name}
        self._stub_bifurcator('create_secret', expected_params, response, error_code=error_code)

    def stub_delete_secret(self, secret_name, error_code=None):
        if False:
            while True:
                i = 10
        expected_params = {'SecretId': secret_name, 'ForceDeleteWithoutRecovery': True}
        self._stub_bifurcator('delete_secret', expected_params, error_code=error_code)

    def stub_describe_secret(self, name, error_code=None):
        if False:
            print('Hello World!')
        expected_params = {'SecretId': name}
        response = {}
        self._stub_bifurcator('describe_secret', expected_params, response, error_code=error_code)

    def stub_get_secret_value(self, name, stage, secret_value, error_code=None):
        if False:
            for i in range(10):
                print('nop')
        expected_params = {'SecretId': name}
        if stage is not None:
            expected_params['VersionStage'] = stage
        response = {}
        if isinstance(secret_value, str):
            response['SecretString'] = secret_value
        elif isinstance(secret_value, bytes):
            response['SecretBinary'] = secret_value
        self._stub_bifurcator('get_secret_value', expected_params, response, error_code=error_code)

    def stub_get_random_password(self, pw_length, password, error_code=None):
        if False:
            while True:
                i = 10
        expected_params = {'PasswordLength': pw_length}
        response = {'RandomPassword': password}
        self._stub_bifurcator('get_random_password', expected_params, response, error_code=error_code)

    def stub_put_secret_value(self, name, secret_value, stages, error_code=None):
        if False:
            for i in range(10):
                print('nop')
        expected_params = {'SecretId': name}
        if isinstance(secret_value, str):
            expected_params['SecretString'] = secret_value
        elif isinstance(secret_value, bytes):
            expected_params['SecretBinary'] = secret_value
        if stages is not None:
            expected_params['VersionStages'] = stages
        response = {}
        self._stub_bifurcator('put_secret_value', expected_params, response, error_code=error_code)

    def stub_update_secret_version_stage(self, name, stage, remove_from, move_to, error_code=None):
        if False:
            print('Hello World!')
        expected_params = {'SecretId': name, 'VersionStage': stage, 'RemoveFromVersionId': remove_from, 'MoveToVersionId': move_to}
        response = {}
        self._stub_bifurcator('update_secret_version_stage', expected_params, response, error_code=error_code)

    def stub_list_secrets(self, secrets, error_code=None):
        if False:
            while True:
                i = 10
        expected_params = {}
        response = {'SecretList': secrets}
        self._stub_bifurcator('list_secrets', expected_params, response, error_code=error_code)