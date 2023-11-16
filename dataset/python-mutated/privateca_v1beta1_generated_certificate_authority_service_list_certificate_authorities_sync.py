from google.cloud.security import privateca_v1beta1

def sample_list_certificate_authorities():
    if False:
        print('Hello World!')
    client = privateca_v1beta1.CertificateAuthorityServiceClient()
    request = privateca_v1beta1.ListCertificateAuthoritiesRequest(parent='parent_value')
    page_result = client.list_certificate_authorities(request=request)
    for response in page_result:
        print(response)