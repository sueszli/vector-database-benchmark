"""
Stub functions that are used by the AWS Organizations unit tests.

When tests are run against an actual AWS account, the stubber class does not
set up stubs and passes all calls through to the Boto3 client.
"""
import json
from test_tools.example_stubber import ExampleStubber

class OrganizationsStubber(ExampleStubber):
    """
    A class that implements a variety of stub functions that are used by the
    AWS Organizations unit tests.

    The stubbed functions all expect certain parameters to be passed to them as
    part of the tests, and will raise errors when the actual parameters differ from
    the expected.
    """

    def __init__(self, client, use_stubs=True):
        if False:
            while True:
                i = 10
        '\n        Initializes the object with a specific client and configures it for\n        stubbing or AWS passthrough.\n\n        :param client: A Boto3 Organizations client.\n        :param use_stubs: When True, use stubs to intercept requests. Otherwise,\n                          pass requests through to AWS.\n        '
        super().__init__(client, use_stubs)

    @staticmethod
    def _make_policy_summary(policy):
        if False:
            return 10
        return {'Id': policy['id'], 'Arn': f"arn:aws:organizations::111111111111:policy/{policy['name']}", 'Name': policy['name'], 'Description': policy['description'], 'Type': policy['type'], 'AwsManaged': False}

    def stub_create_policy(self, policy, error_code=None):
        if False:
            print('Hello World!')
        expected_parameters = {'Name': policy['name'], 'Description': policy['description'], 'Content': json.dumps(policy['content']), 'Type': policy['type']}
        response = {'Policy': {'PolicySummary': self._make_policy_summary(policy), 'Content': json.dumps(policy['content'])}}
        self._stub_bifurcator('create_policy', expected_parameters, response, error_code=error_code)

    def stub_list_policies(self, policy_filter, policies, error_code=None):
        if False:
            i = 10
            return i + 15
        expected_parameters = {'Filter': policy_filter}
        response = {'Policies': [self._make_policy_summary(pol) for pol in policies]}
        self._stub_bifurcator('list_policies', expected_parameters, response, error_code=error_code)

    def stub_describe_policy(self, policy, error_code=None):
        if False:
            while True:
                i = 10
        expected_parameters = {'PolicyId': policy['id']}
        response = {'Policy': {'PolicySummary': self._make_policy_summary(policy), 'Content': json.dumps(policy['content'])}}
        self._stub_bifurcator('describe_policy', expected_parameters, response, error_code=error_code)

    def stub_attach_policy(self, policy_id, target_id, error_code=None):
        if False:
            for i in range(10):
                print('nop')
        expected_parameters = {'PolicyId': policy_id, 'TargetId': target_id}
        self._stub_bifurcator('attach_policy', expected_parameters, error_code=error_code)

    def stub_detach_policy(self, policy_id, target_id, error_code=None):
        if False:
            for i in range(10):
                print('nop')
        expected_parameters = {'PolicyId': policy_id, 'TargetId': target_id}
        self._stub_bifurcator('detach_policy', expected_parameters, error_code=error_code)

    def stub_delete_policy(self, policy_id, error_code=None):
        if False:
            i = 10
            return i + 15
        expected_parameters = {'PolicyId': policy_id}
        self._stub_bifurcator('delete_policy', expected_parameters, error_code=error_code)