from google.cloud import metastore_v1alpha

def sample_update_service():
    if False:
        i = 10
        return i + 15
    client = metastore_v1alpha.DataprocMetastoreClient()
    request = metastore_v1alpha.UpdateServiceRequest()
    operation = client.update_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)