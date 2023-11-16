from google.cloud import certificate_manager_v1

def sample_list_certificate_issuance_configs():
    if False:
        for i in range(10):
            print('nop')
    client = certificate_manager_v1.CertificateManagerClient()
    request = certificate_manager_v1.ListCertificateIssuanceConfigsRequest(parent='parent_value')
    page_result = client.list_certificate_issuance_configs(request=request)
    for response in page_result:
        print(response)