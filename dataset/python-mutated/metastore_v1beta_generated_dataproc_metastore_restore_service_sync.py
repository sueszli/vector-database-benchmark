from google.cloud import metastore_v1beta

def sample_restore_service():
    if False:
        i = 10
        return i + 15
    client = metastore_v1beta.DataprocMetastoreClient()
    request = metastore_v1beta.RestoreServiceRequest(service='service_value', backup='backup_value')
    operation = client.restore_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)