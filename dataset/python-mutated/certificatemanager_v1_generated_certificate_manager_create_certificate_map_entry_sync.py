from google.cloud import certificate_manager_v1

def sample_create_certificate_map_entry():
    if False:
        return 10
    client = certificate_manager_v1.CertificateManagerClient()
    certificate_map_entry = certificate_manager_v1.CertificateMapEntry()
    certificate_map_entry.hostname = 'hostname_value'
    request = certificate_manager_v1.CreateCertificateMapEntryRequest(parent='parent_value', certificate_map_entry_id='certificate_map_entry_id_value', certificate_map_entry=certificate_map_entry)
    operation = client.create_certificate_map_entry(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)