from google.cloud import metastore_v1beta

def sample_delete_service():
    if False:
        i = 10
        return i + 15
    client = metastore_v1beta.DataprocMetastoreClient()
    request = metastore_v1beta.DeleteServiceRequest(name='name_value')
    operation = client.delete_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)