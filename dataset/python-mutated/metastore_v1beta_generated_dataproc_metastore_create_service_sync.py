from google.cloud import metastore_v1beta

def sample_create_service():
    if False:
        i = 10
        return i + 15
    client = metastore_v1beta.DataprocMetastoreClient()
    request = metastore_v1beta.CreateServiceRequest(parent='parent_value', service_id='service_id_value')
    operation = client.create_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)