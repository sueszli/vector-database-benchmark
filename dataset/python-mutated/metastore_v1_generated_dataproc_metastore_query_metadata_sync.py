from google.cloud import metastore_v1

def sample_query_metadata():
    if False:
        for i in range(10):
            print('nop')
    client = metastore_v1.DataprocMetastoreClient()
    request = metastore_v1.QueryMetadataRequest(service='service_value', query='query_value')
    operation = client.query_metadata(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)