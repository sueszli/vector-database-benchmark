from google.cloud import certificate_manager_v1

def sample_list_dns_authorizations():
    if False:
        while True:
            i = 10
    client = certificate_manager_v1.CertificateManagerClient()
    request = certificate_manager_v1.ListDnsAuthorizationsRequest(parent='parent_value')
    page_result = client.list_dns_authorizations(request=request)
    for response in page_result:
        print(response)