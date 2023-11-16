from google.cloud import certificate_manager_v1

def sample_get_certificate_issuance_config():
    if False:
        while True:
            i = 10
    client = certificate_manager_v1.CertificateManagerClient()
    request = certificate_manager_v1.GetCertificateIssuanceConfigRequest(name='name_value')
    response = client.get_certificate_issuance_config(request=request)
    print(response)