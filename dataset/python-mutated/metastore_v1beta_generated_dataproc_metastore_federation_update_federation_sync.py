from google.cloud import metastore_v1beta

def sample_update_federation():
    if False:
        return 10
    client = metastore_v1beta.DataprocMetastoreFederationClient()
    request = metastore_v1beta.UpdateFederationRequest()
    operation = client.update_federation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)