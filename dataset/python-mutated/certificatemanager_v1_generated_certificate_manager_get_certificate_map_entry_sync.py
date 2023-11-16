from google.cloud import certificate_manager_v1

def sample_get_certificate_map_entry():
    if False:
        print('Hello World!')
    client = certificate_manager_v1.CertificateManagerClient()
    request = certificate_manager_v1.GetCertificateMapEntryRequest(name='name_value')
    response = client.get_certificate_map_entry(request=request)
    print(response)