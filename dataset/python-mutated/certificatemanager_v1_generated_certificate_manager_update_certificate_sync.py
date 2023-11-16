from google.cloud import certificate_manager_v1

def sample_update_certificate():
    if False:
        for i in range(10):
            print('nop')
    client = certificate_manager_v1.CertificateManagerClient()
    request = certificate_manager_v1.UpdateCertificateRequest()
    operation = client.update_certificate(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)