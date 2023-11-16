from google.cloud.security import privateca_v1beta1

def sample_update_certificate():
    if False:
        i = 10
        return i + 15
    client = privateca_v1beta1.CertificateAuthorityServiceClient()
    certificate = privateca_v1beta1.Certificate()
    certificate.pem_csr = 'pem_csr_value'
    request = privateca_v1beta1.UpdateCertificateRequest(certificate=certificate)
    response = client.update_certificate(request=request)
    print(response)