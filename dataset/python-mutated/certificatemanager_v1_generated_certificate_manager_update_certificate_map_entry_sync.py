from google.cloud import certificate_manager_v1

def sample_update_certificate_map_entry():
    if False:
        for i in range(10):
            print('nop')
    client = certificate_manager_v1.CertificateManagerClient()
    certificate_map_entry = certificate_manager_v1.CertificateMapEntry()
    certificate_map_entry.hostname = 'hostname_value'
    request = certificate_manager_v1.UpdateCertificateMapEntryRequest(certificate_map_entry=certificate_map_entry)
    operation = client.update_certificate_map_entry(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)