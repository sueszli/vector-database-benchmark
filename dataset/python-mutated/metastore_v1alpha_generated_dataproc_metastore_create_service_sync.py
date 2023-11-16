from google.cloud import metastore_v1alpha

def sample_create_service():
    if False:
        return 10
    client = metastore_v1alpha.DataprocMetastoreClient()
    request = metastore_v1alpha.CreateServiceRequest(parent='parent_value', service_id='service_id_value')
    operation = client.create_service(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)