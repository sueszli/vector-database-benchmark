from google.cloud import metastore_v1

def sample_create_service():
    if False:
        while True:
            i = 10
    client = metastore_v1.DataprocMetastoreClient()
    request = metastore_v1.CreateServiceRequest(parent='parent_value', service_id='service_id_value')
    operation = client.create_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)