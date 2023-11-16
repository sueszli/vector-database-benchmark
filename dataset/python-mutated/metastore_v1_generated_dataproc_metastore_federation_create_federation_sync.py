from google.cloud import metastore_v1

def sample_create_federation():
    if False:
        for i in range(10):
            print('nop')
    client = metastore_v1.DataprocMetastoreFederationClient()
    request = metastore_v1.CreateFederationRequest(parent='parent_value', federation_id='federation_id_value')
    operation = client.create_federation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)