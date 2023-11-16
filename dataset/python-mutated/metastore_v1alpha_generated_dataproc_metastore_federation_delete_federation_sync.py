from google.cloud import metastore_v1alpha

def sample_delete_federation():
    if False:
        for i in range(10):
            print('nop')
    client = metastore_v1alpha.DataprocMetastoreFederationClient()
    request = metastore_v1alpha.DeleteFederationRequest(name='name_value')
    operation = client.delete_federation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)