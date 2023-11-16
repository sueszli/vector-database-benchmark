from collections import OrderedDict
import pytest
from chalice.awsclient import TypedAWSClient

@pytest.mark.parametrize('service,region,endpoint', [('sns', 'us-east-1', OrderedDict([('partition', 'aws'), ('endpointName', 'us-east-1'), ('protocols', ['http', 'https']), ('hostname', 'sns.us-east-1.amazonaws.com'), ('signatureVersions', ['v4']), ('dnsSuffix', 'amazonaws.com')])), ('sqs', 'cn-north-1', OrderedDict([('partition', 'aws-cn'), ('endpointName', 'cn-north-1'), ('protocols', ['http', 'https']), ('sslCommonName', 'cn-north-1.queue.amazonaws.com.cn'), ('hostname', 'sqs.cn-north-1.amazonaws.com.cn'), ('signatureVersions', ['v4']), ('dnsSuffix', 'amazonaws.com.cn')])), ('dynamodb', 'mars-west-1', None)])
def test_resolve_endpoint(stubbed_session, service, region, endpoint):
    if False:
        return 10
    awsclient = TypedAWSClient(stubbed_session)
    if endpoint is None:
        assert awsclient.resolve_endpoint(service, region) is None
    else:
        assert endpoint.items() <= awsclient.resolve_endpoint(service, region).items()

@pytest.mark.parametrize('arn,endpoint', [('arn:aws:sns:us-east-1:123456:MyTopic', OrderedDict([('partition', 'aws'), ('endpointName', 'us-east-1'), ('protocols', ['http', 'https']), ('hostname', 'sns.us-east-1.amazonaws.com'), ('signatureVersions', ['v4']), ('dnsSuffix', 'amazonaws.com')])), ('arn:aws-cn:sqs:cn-north-1:444455556666:queue1', OrderedDict([('partition', 'aws-cn'), ('endpointName', 'cn-north-1'), ('protocols', ['http', 'https']), ('sslCommonName', 'cn-north-1.queue.amazonaws.com.cn'), ('hostname', 'sqs.cn-north-1.amazonaws.com.cn'), ('signatureVersions', ['v4']), ('dnsSuffix', 'amazonaws.com.cn')])), ('arn:aws:dynamodb:mars-west-1:123456:table/MyTable', None)])
def test_endpoint_from_arn(stubbed_session, arn, endpoint):
    if False:
        return 10
    awsclient = TypedAWSClient(stubbed_session)
    if endpoint is None:
        assert awsclient.endpoint_from_arn(arn) is None
    else:
        assert endpoint.items() <= awsclient.endpoint_from_arn(arn).items()

@pytest.mark.parametrize('service,region,dns_suffix', [('sns', 'us-east-1', 'amazonaws.com'), ('sns', 'cn-north-1', 'amazonaws.com.cn'), ('dynamodb', 'mars-west-1', 'amazonaws.com')])
def test_endpoint_dns_suffix(stubbed_session, service, region, dns_suffix):
    if False:
        i = 10
        return i + 15
    awsclient = TypedAWSClient(stubbed_session)
    assert dns_suffix == awsclient.endpoint_dns_suffix(service, region)

@pytest.mark.parametrize('arn,dns_suffix', [('arn:aws:sns:us-east-1:123456:MyTopic', 'amazonaws.com'), ('arn:aws-cn:sqs:cn-north-1:444455556666:queue1', 'amazonaws.com.cn'), ('arn:aws:dynamodb:mars-west-1:123456:table/MyTable', 'amazonaws.com')])
def test_endpoint_dns_suffix_from_arn(stubbed_session, arn, dns_suffix):
    if False:
        print('Hello World!')
    awsclient = TypedAWSClient(stubbed_session)
    assert dns_suffix == awsclient.endpoint_dns_suffix_from_arn(arn)

class TestServicePrincipal(object):

    @pytest.fixture
    def region(self):
        if False:
            i = 10
            return i + 15
        return 'bermuda-triangle-42'

    @pytest.fixture
    def url_suffix(self):
        if False:
            print('Hello World!')
        return '.nowhere.null'

    @pytest.fixture
    def non_iso_suffixes(self):
        if False:
            while True:
                i = 10
        return ['', '.amazonaws.com', '.amazonaws.com.cn']

    @pytest.fixture
    def awsclient(self, stubbed_session):
        if False:
            print('Hello World!')
        return TypedAWSClient(stubbed_session)

    def test_unmatched_service(self, awsclient):
        if False:
            i = 10
            return i + 15
        assert awsclient.service_principal('taco.magic.food.com', 'us-east-1', 'amazonaws.com') == 'taco.magic.food.com'

    def test_defaults(self, awsclient):
        if False:
            while True:
                i = 10
        assert awsclient.service_principal('lambda') == 'lambda.amazonaws.com'

    def test_states(self, awsclient, region, url_suffix, non_iso_suffixes):
        if False:
            print('Hello World!')
        services = ['states']
        for suffix in non_iso_suffixes:
            for service in services:
                assert awsclient.service_principal('{}{}'.format(service, suffix), region, url_suffix) == '{}.{}.amazonaws.com'.format(service, region)

    def test_codedeploy_and_logs(self, awsclient, region, url_suffix, non_iso_suffixes):
        if False:
            i = 10
            return i + 15
        services = ['codedeploy', 'logs']
        for suffix in non_iso_suffixes:
            for service in services:
                assert awsclient.service_principal('{}{}'.format(service, suffix), region, url_suffix) == '{}.{}.{}'.format(service, region, url_suffix)

    def test_ec2(self, awsclient, region, url_suffix, non_iso_suffixes):
        if False:
            return 10
        services = ['ec2']
        for suffix in non_iso_suffixes:
            for service in services:
                assert awsclient.service_principal('{}{}'.format(service, suffix), region, url_suffix) == '{}.{}'.format(service, url_suffix)

    def test_others(self, awsclient, region, url_suffix, non_iso_suffixes):
        if False:
            i = 10
            return i + 15
        services = ['autoscaling', 'lambda', 'events', 'sns', 'sqs', 'foo-service']
        for suffix in non_iso_suffixes:
            for service in services:
                assert awsclient.service_principal('{}{}'.format(service, suffix), region, url_suffix) == '{}.amazonaws.com'.format(service)

    def test_local_suffix(self, awsclient, region, url_suffix):
        if False:
            for i in range(10):
                print('nop')
        assert awsclient.service_principal('foo-service.local', region, url_suffix) == 'foo-service.local'

    def test_states_iso(self, awsclient):
        if False:
            return 10
        assert awsclient.service_principal('states.amazonaws.com', 'us-iso-east-1', 'c2s.ic.gov') == 'states.amazonaws.com'

    def test_states_isob(self, awsclient):
        if False:
            print('Hello World!')
        assert awsclient.service_principal('states.amazonaws.com', 'us-isob-east-1', 'sc2s.sgov.gov') == 'states.amazonaws.com'

    def test_iso_exceptions(self, awsclient):
        if False:
            while True:
                i = 10
        services = ['cloudhsm', 'config', 'workspaces']
        for service in services:
            assert awsclient.service_principal('{}.amazonaws.com'.format(service), 'us-iso-east-1', 'c2s.ic.gov') == '{}.c2s.ic.gov'.format(service)