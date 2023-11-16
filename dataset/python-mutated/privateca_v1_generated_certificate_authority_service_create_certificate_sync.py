from google.cloud.security import privateca_v1

def sample_create_certificate():
    if False:
        while True:
            i = 10
    client = privateca_v1.CertificateAuthorityServiceClient()
    certificate = privateca_v1.Certificate()
    certificate.pem_csr = 'pem_csr_value'
    request = privateca_v1.CreateCertificateRequest(parent='parent_value', certificate=certificate)
    response = client.create_certificate(request=request)
    print(response)