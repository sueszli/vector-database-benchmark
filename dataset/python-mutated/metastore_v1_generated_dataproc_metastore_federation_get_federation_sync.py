from google.cloud import metastore_v1

def sample_get_federation():
    if False:
        i = 10
        return i + 15
    client = metastore_v1.DataprocMetastoreFederationClient()
    request = metastore_v1.GetFederationRequest(name='name_value')
    response = client.get_federation(request=request)
    print(response)