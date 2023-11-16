from google.cloud import metastore_v1beta

def sample_get_federation():
    if False:
        print('Hello World!')
    client = metastore_v1beta.DataprocMetastoreFederationClient()
    request = metastore_v1beta.GetFederationRequest(name='name_value')
    response = client.get_federation(request=request)
    print(response)