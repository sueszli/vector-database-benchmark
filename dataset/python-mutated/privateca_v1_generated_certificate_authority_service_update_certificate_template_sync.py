from google.cloud.security import privateca_v1

def sample_update_certificate_template():
    if False:
        while True:
            i = 10
    client = privateca_v1.CertificateAuthorityServiceClient()
    request = privateca_v1.UpdateCertificateTemplateRequest()
    operation = client.update_certificate_template(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)