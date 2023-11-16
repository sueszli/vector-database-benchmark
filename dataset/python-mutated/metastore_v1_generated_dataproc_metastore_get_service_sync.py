from google.cloud import metastore_v1

def sample_get_service():
    if False:
        return 10
    client = metastore_v1.DataprocMetastoreClient()
    request = metastore_v1.GetServiceRequest(name='name_value')
    response = client.get_service(request=request)
    print(response)