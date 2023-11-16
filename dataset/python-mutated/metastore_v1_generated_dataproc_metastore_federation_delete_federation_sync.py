from google.cloud import metastore_v1

def sample_delete_federation():
    if False:
        while True:
            i = 10
    client = metastore_v1.DataprocMetastoreFederationClient()
    request = metastore_v1.DeleteFederationRequest(name='name_value')
    operation = client.delete_federation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)