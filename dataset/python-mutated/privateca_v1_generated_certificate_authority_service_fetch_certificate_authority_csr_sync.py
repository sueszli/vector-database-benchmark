from google.cloud.security import privateca_v1

def sample_fetch_certificate_authority_csr():
    if False:
        print('Hello World!')
    client = privateca_v1.CertificateAuthorityServiceClient()
    request = privateca_v1.FetchCertificateAuthorityCsrRequest(name='name_value')
    response = client.fetch_certificate_authority_csr(request=request)
    print(response)