from google.cloud import metastore_v1

def sample_delete_service():
    if False:
        for i in range(10):
            print('nop')
    client = metastore_v1.DataprocMetastoreClient()
    request = metastore_v1.DeleteServiceRequest(name='name_value')
    operation = client.delete_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)