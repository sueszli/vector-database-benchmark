from google.cloud.security import privateca_v1beta1

def sample_list_certificates():
    if False:
        return 10
    client = privateca_v1beta1.CertificateAuthorityServiceClient()
    request = privateca_v1beta1.ListCertificatesRequest(parent='parent_value')
    page_result = client.list_certificates(request=request)
    for response in page_result:
        print(response)