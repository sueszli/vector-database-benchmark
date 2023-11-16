from google.cloud import certificate_manager_v1

def sample_get_certificate():
    if False:
        while True:
            i = 10
    client = certificate_manager_v1.CertificateManagerClient()
    request = certificate_manager_v1.GetCertificateRequest(name='name_value')
    response = client.get_certificate(request=request)
    print(response)