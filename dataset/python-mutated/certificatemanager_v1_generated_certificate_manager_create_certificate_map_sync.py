from google.cloud import certificate_manager_v1

def sample_create_certificate_map():
    if False:
        for i in range(10):
            print('nop')
    client = certificate_manager_v1.CertificateManagerClient()
    request = certificate_manager_v1.CreateCertificateMapRequest(parent='parent_value', certificate_map_id='certificate_map_id_value')
    operation = client.create_certificate_map(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)