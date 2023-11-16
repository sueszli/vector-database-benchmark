from google.cloud.security import privateca_v1beta1

def sample_update_certificate_revocation_list():
    if False:
        i = 10
        return i + 15
    client = privateca_v1beta1.CertificateAuthorityServiceClient()
    request = privateca_v1beta1.UpdateCertificateRevocationListRequest()
    operation = client.update_certificate_revocation_list(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)