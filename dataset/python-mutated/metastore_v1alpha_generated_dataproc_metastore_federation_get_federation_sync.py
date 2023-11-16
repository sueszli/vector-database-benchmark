from google.cloud import metastore_v1alpha

def sample_get_federation():
    if False:
        while True:
            i = 10
    client = metastore_v1alpha.DataprocMetastoreFederationClient()
    request = metastore_v1alpha.GetFederationRequest(name='name_value')
    response = client.get_federation(request=request)
    print(response)