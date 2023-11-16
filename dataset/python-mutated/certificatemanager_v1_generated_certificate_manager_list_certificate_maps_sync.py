from google.cloud import certificate_manager_v1

def sample_list_certificate_maps():
    if False:
        while True:
            i = 10
    client = certificate_manager_v1.CertificateManagerClient()
    request = certificate_manager_v1.ListCertificateMapsRequest(parent='parent_value')
    page_result = client.list_certificate_maps(request=request)
    for response in page_result:
        print(response)