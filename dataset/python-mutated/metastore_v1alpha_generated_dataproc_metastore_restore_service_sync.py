from google.cloud import metastore_v1alpha

def sample_restore_service():
    if False:
        return 10
    client = metastore_v1alpha.DataprocMetastoreClient()
    request = metastore_v1alpha.RestoreServiceRequest(service='service_value', backup='backup_value')
    operation = client.restore_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)