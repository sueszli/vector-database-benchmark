from google.cloud.security import privateca_v1

def sample_list_ca_pools():
    if False:
        return 10
    client = privateca_v1.CertificateAuthorityServiceClient()
    request = privateca_v1.ListCaPoolsRequest(parent='parent_value')
    page_result = client.list_ca_pools(request=request)
    for response in page_result:
        print(response)