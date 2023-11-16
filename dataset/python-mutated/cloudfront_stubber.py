"""
Stub functions that are used by the Amazon CloudFront unit tests.

When tests are run against an actual AWS account, the stubber class does not
set up stubs and passes all calls through to the Boto3 client.
"""
import datetime
from test_tools.example_stubber import ExampleStubber

class CloudFrontStubber(ExampleStubber):
    """
    A class that implements stub functions used by CloudFront unit tests.

    The stubbed functions expect certain parameters to be passed to them as
    part of the tests, and raise errors if the parameters are not as expected.
    """

    def __init__(self, client, use_stubs=True):
        if False:
            i = 10
            return i + 15
        '\n        Initializes the object with a specific client and configures it for\n        stubbing or AWS passthrough.\n\n        :param client: A Boto3 CloudFront client.\n        :param use_stubs: When True, use stubs to intercept requests. Otherwise,\n                          pass requests through to AWS.\n        '
        super().__init__(client, use_stubs)

    def stub_list_distributions(self, distribs, error_code=None):
        if False:
            print('Hello World!')
        expected_params = {}
        response = {'DistributionList': {'Marker': 'marker', 'MaxItems': 100, 'IsTruncated': False, 'Quantity': len(distribs), 'Items': [{'ARN': f'arn:aws:cloudfront::123456789012:distribution/{index}', 'Status': 'Deployed', 'LastModifiedTime': datetime.datetime.now(), 'Aliases': {'Quantity': 0}, 'Origins': {'Quantity': 0, 'Items': [{'Id': 'test-id', 'DomainName': 'test'}]}, 'DefaultCacheBehavior': {'TargetOriginId': '', 'ViewerProtocolPolicy': ''}, 'CacheBehaviors': {'Quantity': 0}, 'CustomErrorResponses': {'Quantity': 0}, 'Comment': 'Testing!', 'PriceClass': 'PriceClass_All', 'Enabled': True, 'Restrictions': {'GeoRestriction': {'Quantity': 0, 'RestrictionType': ''}}, 'WebACLId': '', 'HttpVersion': 'http2', 'IsIPV6Enabled': True, 'DomainName': distrib['name'], 'Id': distrib['id'], 'ViewerCertificate': {'CertificateSource': distrib['cert_source'], 'Certificate': distrib['cert']}, 'Staging': False} for (index, distrib) in enumerate(distribs)]}}
        self._stub_bifurcator('list_distributions', expected_params, response, error_code=error_code)

    def stub_get_distribution_config(self, distrib_id, comment, etag, error_code=None):
        if False:
            while True:
                i = 10
        expected_params = {'Id': distrib_id}
        response = {'DistributionConfig': {'CallerReference': 'test', 'Origins': {'Quantity': 0, 'Items': [{'Id': 'test-id', 'DomainName': 'test'}]}, 'DefaultCacheBehavior': {'TargetOriginId': '', 'ViewerProtocolPolicy': ''}, 'Enabled': True, 'Comment': comment}, 'ETag': etag}
        self._stub_bifurcator('get_distribution_config', expected_params, response, error_code=error_code)

    def stub_update_distribution(self, distrib_id, comment, etag, error_code=None):
        if False:
            for i in range(10):
                print('nop')
        expected_params = {'Id': distrib_id, 'DistributionConfig': {'CallerReference': 'test', 'Origins': {'Quantity': 0, 'Items': [{'Id': 'test-id', 'DomainName': 'test'}]}, 'DefaultCacheBehavior': {'TargetOriginId': '', 'ViewerProtocolPolicy': ''}, 'Enabled': True, 'Comment': comment}, 'IfMatch': etag}
        response = {}
        self._stub_bifurcator('update_distribution', expected_params, response, error_code=error_code)