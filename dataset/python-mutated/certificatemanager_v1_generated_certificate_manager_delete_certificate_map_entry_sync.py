from google.cloud import certificate_manager_v1

def sample_delete_certificate_map_entry():
    if False:
        while True:
            i = 10
    client = certificate_manager_v1.CertificateManagerClient()
    request = certificate_manager_v1.DeleteCertificateMapEntryRequest(name='name_value')
    operation = client.delete_certificate_map_entry(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)