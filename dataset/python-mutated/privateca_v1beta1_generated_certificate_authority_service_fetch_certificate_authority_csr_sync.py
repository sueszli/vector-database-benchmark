from google.cloud.security import privateca_v1beta1

def sample_fetch_certificate_authority_csr():
    if False:
        while True:
            i = 10
    client = privateca_v1beta1.CertificateAuthorityServiceClient()
    request = privateca_v1beta1.FetchCertificateAuthorityCsrRequest(name='name_value')
    response = client.fetch_certificate_authority_csr(request=request)
    print(response)