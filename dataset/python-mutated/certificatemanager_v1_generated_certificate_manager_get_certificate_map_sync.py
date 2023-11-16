from google.cloud import certificate_manager_v1

def sample_get_certificate_map():
    if False:
        i = 10
        return i + 15
    client = certificate_manager_v1.CertificateManagerClient()
    request = certificate_manager_v1.GetCertificateMapRequest(name='name_value')
    response = client.get_certificate_map(request=request)
    print(response)