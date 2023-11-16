from google.cloud.security import privateca_v1

def sample_update_certificate():
    if False:
        print('Hello World!')
    client = privateca_v1.CertificateAuthorityServiceClient()
    certificate = privateca_v1.Certificate()
    certificate.pem_csr = 'pem_csr_value'
    request = privateca_v1.UpdateCertificateRequest(certificate=certificate)
    response = client.update_certificate(request=request)
    print(response)