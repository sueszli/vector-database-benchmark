from google.cloud import metastore_v1beta

def sample_query_metadata():
    if False:
        while True:
            i = 10
    client = metastore_v1beta.DataprocMetastoreClient()
    request = metastore_v1beta.QueryMetadataRequest(service='service_value', query='query_value')
    operation = client.query_metadata(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)