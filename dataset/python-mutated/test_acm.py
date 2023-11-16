from localstack.testing.pytest import markers
from localstack.utils.common import short_uid
TEST_TEMPLATE = '\nResources:\n  cert1:\n    Type: "AWS::CertificateManager::Certificate"\n    Properties:\n      DomainName: "{{domain}}"\n      DomainValidationOptions:\n        - DomainName: "{{domain}}"\n          HostedZoneId: zone123  # using dummy ID for now\n      ValidationMethod: DNS\nOutputs:\n  Cert:\n    Value: !Ref cert1\n'

@markers.aws.only_localstack
def test_cfn_acm_certificate(deploy_cfn_template, aws_client):
    if False:
        while True:
            i = 10
    domain = f'domain-{short_uid()}.com'
    deploy_cfn_template(template=TEST_TEMPLATE, template_mapping={'domain': domain})
    result = aws_client.acm.list_certificates()['CertificateSummaryList']
    result = [cert for cert in result if cert['DomainName'] == domain]
    assert result