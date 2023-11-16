from google.cloud import metastore_v1beta

def sample_get_service():
    if False:
        for i in range(10):
            print('nop')
    client = metastore_v1beta.DataprocMetastoreClient()
    request = metastore_v1beta.GetServiceRequest(name='name_value')
    response = client.get_service(request=request)
    print(response)