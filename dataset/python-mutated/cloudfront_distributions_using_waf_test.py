from unittest import mock
from moto.core import DEFAULT_ACCOUNT_ID
from prowler.providers.aws.services.cloudfront.cloudfront_service import Distribution
DISTRIBUTION_ID = 'E27LVI50CSW06W'
DISTRIBUTION_ARN = f'arn:aws:cloudfront::{DEFAULT_ACCOUNT_ID}:distribution/{DISTRIBUTION_ID}'
REGION = 'eu-west-1'

class Test_cloudfront_distributions_using_waf:

    def test_no_distributions(self):
        if False:
            return 10
        cloudfront_client = mock.MagicMock
        cloudfront_client.distributions = {}
        with mock.patch('prowler.providers.aws.services.cloudfront.cloudfront_service.CloudFront', new=cloudfront_client):
            from prowler.providers.aws.services.cloudfront.cloudfront_distributions_using_waf.cloudfront_distributions_using_waf import cloudfront_distributions_using_waf
            check = cloudfront_distributions_using_waf()
            result = check.execute()
            assert len(result) == 0

    def test_one_distribution_waf(self):
        if False:
            print('Hello World!')
        wef_acl_id = 'TEST-WAF-ACL'
        cloudfront_client = mock.MagicMock
        cloudfront_client.distributions = {'DISTRIBUTION_ID': Distribution(arn=DISTRIBUTION_ARN, id=DISTRIBUTION_ID, region=REGION, web_acl_id=wef_acl_id, origins=[])}
        with mock.patch('prowler.providers.aws.services.cloudfront.cloudfront_service.CloudFront', new=cloudfront_client):
            from prowler.providers.aws.services.cloudfront.cloudfront_distributions_using_waf.cloudfront_distributions_using_waf import cloudfront_distributions_using_waf
            check = cloudfront_distributions_using_waf()
            result = check.execute()
            assert len(result) == 1
            assert result[0].region == REGION
            assert result[0].resource_arn == DISTRIBUTION_ARN
            assert result[0].resource_id == DISTRIBUTION_ID
            assert result[0].status == 'PASS'
            assert result[0].status_extended == f'CloudFront Distribution {DISTRIBUTION_ID} is using AWS WAF web ACL {wef_acl_id}.'
            assert result[0].resource_tags == []

    def test_one_distribution_no_waf(self):
        if False:
            i = 10
            return i + 15
        cloudfront_client = mock.MagicMock
        cloudfront_client.distributions = {'DISTRIBUTION_ID': Distribution(arn=DISTRIBUTION_ARN, id=DISTRIBUTION_ID, region=REGION, origins=[])}
        with mock.patch('prowler.providers.aws.services.cloudfront.cloudfront_service.CloudFront', new=cloudfront_client):
            from prowler.providers.aws.services.cloudfront.cloudfront_distributions_using_waf.cloudfront_distributions_using_waf import cloudfront_distributions_using_waf
            check = cloudfront_distributions_using_waf()
            result = check.execute()
            assert len(result) == 1
            assert result[0].region == REGION
            assert result[0].resource_arn == DISTRIBUTION_ARN
            assert result[0].resource_id == DISTRIBUTION_ID
            assert result[0].status == 'FAIL'
            assert result[0].status_extended == f'CloudFront Distribution {DISTRIBUTION_ID} is not using AWS WAF web ACL.'
            assert result[0].resource_tags == []