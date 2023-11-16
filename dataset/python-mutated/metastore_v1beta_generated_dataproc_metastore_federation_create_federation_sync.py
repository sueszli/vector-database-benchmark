from google.cloud import metastore_v1beta

def sample_create_federation():
    if False:
        for i in range(10):
            print('nop')
    client = metastore_v1beta.DataprocMetastoreFederationClient()
    request = metastore_v1beta.CreateFederationRequest(parent='parent_value', federation_id='federation_id_value')
    operation = client.create_federation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)