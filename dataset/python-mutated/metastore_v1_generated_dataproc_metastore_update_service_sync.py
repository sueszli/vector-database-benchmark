from google.cloud import metastore_v1

def sample_update_service():
    if False:
        for i in range(10):
            print('nop')
    client = metastore_v1.DataprocMetastoreClient()
    request = metastore_v1.UpdateServiceRequest()
    operation = client.update_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)