from google.cloud import certificate_manager_v1

def sample_update_dns_authorization():
    if False:
        return 10
    client = certificate_manager_v1.CertificateManagerClient()
    dns_authorization = certificate_manager_v1.DnsAuthorization()
    dns_authorization.domain = 'domain_value'
    request = certificate_manager_v1.UpdateDnsAuthorizationRequest(dns_authorization=dns_authorization)
    operation = client.update_dns_authorization(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)