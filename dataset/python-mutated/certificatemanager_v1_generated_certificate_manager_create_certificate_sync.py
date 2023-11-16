from google.cloud import certificate_manager_v1

def sample_create_certificate():
    if False:
        for i in range(10):
            print('nop')
    client = certificate_manager_v1.CertificateManagerClient()
    request = certificate_manager_v1.CreateCertificateRequest(parent='parent_value', certificate_id='certificate_id_value')
    operation = client.create_certificate(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)