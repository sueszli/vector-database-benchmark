"""
Stub functions that are used by the Amazon EC2 Bedrock unit tests.
"""
from test_tools.example_stubber import ExampleStubber

class BedrockStubber(ExampleStubber):
    """
    A class that implements stub functions used by Amazon Bedrock unit tests.
    """

    def __init__(self, client, use_stubs=True):
        if False:
            while True:
                i = 10
        '\n        Initializes the object with a specific client and configures it for\n        stubbing or AWS passthrough.\n\n        :param client: A Boto3 Amazon Bedrock client.\n        :param use_stubs: When True, uses stubs to intercept requests. Otherwise,\n                          passes requests through to AWS.\n        '
        super().__init__(client, use_stubs)

    def stub_list_foundation_models(self, models, error_code=None):
        if False:
            print('Hello World!')
        expected_params = {}
        response = {'modelSummaries': models}
        self._stub_bifurcator('list_foundation_models', expected_params, response, error_code=error_code)