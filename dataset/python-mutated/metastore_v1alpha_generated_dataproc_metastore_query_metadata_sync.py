from google.cloud import metastore_v1alpha

def sample_query_metadata():
    if False:
        print('Hello World!')
    client = metastore_v1alpha.DataprocMetastoreClient()
    request = metastore_v1alpha.QueryMetadataRequest(service='service_value', query='query_value')
    operation = client.query_metadata(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)