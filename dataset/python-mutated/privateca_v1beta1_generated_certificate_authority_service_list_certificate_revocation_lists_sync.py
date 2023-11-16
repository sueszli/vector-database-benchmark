from google.cloud.security import privateca_v1beta1

def sample_list_certificate_revocation_lists():
    if False:
        return 10
    client = privateca_v1beta1.CertificateAuthorityServiceClient()
    request = privateca_v1beta1.ListCertificateRevocationListsRequest(parent='parent_value')
    page_result = client.list_certificate_revocation_lists(request=request)
    for response in page_result:
        print(response)