"""
Stub functions that are used by the Amazon API Gateway Management API unit tests.

When tests are run against an actual AWS account, the stubber class does not
set up stubs and passes all calls through to the Boto3 client.
"""
from test_tools.example_stubber import ExampleStubber

class ApiGatewayManagementApiStubber(ExampleStubber):
    """
    A class that implements a variety of stub functions that are used by the
    Amazon API Gateway Management API unit tests.

    The stubbed functions all expect certain parameters to be passed to them as
    part of the tests, and will raise errors when the actual parameters differ from
    the expected.
    """

    def __init__(self, client, use_stubs=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initializes the object with a specific client and configures it for\n        stubbing or AWS passthrough.\n\n        :param client: A Boto3 API Gateway Management API client.\n        :param use_stubs: When True, use stubs to intercept requests. Otherwise,\n                          pass requests through to AWS.\n        '
        super().__init__(client, use_stubs)

    def stub_post_to_connection(self, message, connection_id, error_code=None):
        if False:
            for i in range(10):
                print('nop')
        expected_params = {'Data': message, 'ConnectionId': connection_id}
        response = {}
        self._stub_bifurcator('post_to_connection', expected_params, response, error_code=error_code)