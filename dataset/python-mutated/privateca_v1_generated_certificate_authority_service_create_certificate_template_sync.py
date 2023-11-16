from google.cloud.security import privateca_v1

def sample_create_certificate_template():
    if False:
        for i in range(10):
            print('nop')
    client = privateca_v1.CertificateAuthorityServiceClient()
    request = privateca_v1.CreateCertificateTemplateRequest(parent='parent_value', certificate_template_id='certificate_template_id_value')
    operation = client.create_certificate_template(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)