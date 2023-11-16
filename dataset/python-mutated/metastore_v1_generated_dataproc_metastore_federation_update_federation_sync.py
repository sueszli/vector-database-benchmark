from google.cloud import metastore_v1

def sample_update_federation():
    if False:
        print('Hello World!')
    client = metastore_v1.DataprocMetastoreFederationClient()
    request = metastore_v1.UpdateFederationRequest()
    operation = client.update_federation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)