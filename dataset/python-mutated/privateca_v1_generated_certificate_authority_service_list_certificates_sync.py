from google.cloud.security import privateca_v1

def sample_list_certificates():
    if False:
        for i in range(10):
            print('nop')
    client = privateca_v1.CertificateAuthorityServiceClient()
    request = privateca_v1.ListCertificatesRequest(parent='parent_value')
    page_result = client.list_certificates(request=request)
    for response in page_result:
        print(response)