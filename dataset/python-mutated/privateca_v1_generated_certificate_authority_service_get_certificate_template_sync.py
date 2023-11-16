from google.cloud.security import privateca_v1

def sample_get_certificate_template():
    if False:
        while True:
            i = 10
    client = privateca_v1.CertificateAuthorityServiceClient()
    request = privateca_v1.GetCertificateTemplateRequest(name='name_value')
    response = client.get_certificate_template(request=request)
    print(response)