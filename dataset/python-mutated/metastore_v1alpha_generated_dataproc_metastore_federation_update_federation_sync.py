from google.cloud import metastore_v1alpha

def sample_update_federation():
    if False:
        while True:
            i = 10
    client = metastore_v1alpha.DataprocMetastoreFederationClient()
    request = metastore_v1alpha.UpdateFederationRequest()
    operation = client.update_federation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)