from google.cloud import certificate_manager_v1

def sample_create_dns_authorization():
    if False:
        for i in range(10):
            print('nop')
    client = certificate_manager_v1.CertificateManagerClient()
    dns_authorization = certificate_manager_v1.DnsAuthorization()
    dns_authorization.domain = 'domain_value'
    request = certificate_manager_v1.CreateDnsAuthorizationRequest(parent='parent_value', dns_authorization_id='dns_authorization_id_value', dns_authorization=dns_authorization)
    operation = client.create_dns_authorization(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)