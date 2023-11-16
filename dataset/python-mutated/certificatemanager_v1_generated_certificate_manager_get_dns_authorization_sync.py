from google.cloud import certificate_manager_v1

def sample_get_dns_authorization():
    if False:
        return 10
    client = certificate_manager_v1.CertificateManagerClient()
    request = certificate_manager_v1.GetDnsAuthorizationRequest(name='name_value')
    response = client.get_dns_authorization(request=request)
    print(response)