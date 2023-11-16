from google.cloud import metastore_v1beta

def sample_delete_federation():
    if False:
        print('Hello World!')
    client = metastore_v1beta.DataprocMetastoreFederationClient()
    request = metastore_v1beta.DeleteFederationRequest(name='name_value')
    operation = client.delete_federation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)