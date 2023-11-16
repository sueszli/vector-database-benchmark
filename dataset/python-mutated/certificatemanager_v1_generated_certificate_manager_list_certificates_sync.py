from google.cloud import certificate_manager_v1

def sample_list_certificates():
    if False:
        i = 10
        return i + 15
    client = certificate_manager_v1.CertificateManagerClient()
    request = certificate_manager_v1.ListCertificatesRequest(parent='parent_value')
    page_result = client.list_certificates(request=request)
    for response in page_result:
        print(response)