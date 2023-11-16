from google.cloud.security import privateca_v1

def sample_list_certificate_templates():
    if False:
        print('Hello World!')
    client = privateca_v1.CertificateAuthorityServiceClient()
    request = privateca_v1.ListCertificateTemplatesRequest(parent='parent_value')
    page_result = client.list_certificate_templates(request=request)
    for response in page_result:
        print(response)