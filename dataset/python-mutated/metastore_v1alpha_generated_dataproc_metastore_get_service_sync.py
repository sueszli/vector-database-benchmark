from google.cloud import metastore_v1alpha

def sample_get_service():
    if False:
        i = 10
        return i + 15
    client = metastore_v1alpha.DataprocMetastoreClient()
    request = metastore_v1alpha.GetServiceRequest(name='name_value')
    response = client.get_service(request=request)
    print(response)