from google.cloud.security import privateca_v1

def sample_list_certificate_revocation_lists():
    if False:
        for i in range(10):
            print('nop')
    client = privateca_v1.CertificateAuthorityServiceClient()
    request = privateca_v1.ListCertificateRevocationListsRequest(parent='parent_value')
    page_result = client.list_certificate_revocation_lists(request=request)
    for response in page_result:
        print(response)