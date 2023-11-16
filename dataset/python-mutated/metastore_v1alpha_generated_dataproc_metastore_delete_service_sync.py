from google.cloud import metastore_v1alpha

def sample_delete_service():
    if False:
        return 10
    client = metastore_v1alpha.DataprocMetastoreClient()
    request = metastore_v1alpha.DeleteServiceRequest(name='name_value')
    operation = client.delete_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)