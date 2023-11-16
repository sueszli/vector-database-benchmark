from google.cloud.security import privateca_v1

def sample_delete_certificate_template():
    if False:
        i = 10
        return i + 15
    client = privateca_v1.CertificateAuthorityServiceClient()
    request = privateca_v1.DeleteCertificateTemplateRequest(name='name_value')
    operation = client.delete_certificate_template(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)