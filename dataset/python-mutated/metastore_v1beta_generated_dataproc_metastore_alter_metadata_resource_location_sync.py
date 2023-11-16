from google.cloud import metastore_v1beta

def sample_alter_metadata_resource_location():
    if False:
        return 10
    client = metastore_v1beta.DataprocMetastoreClient()
    request = metastore_v1beta.AlterMetadataResourceLocationRequest(service='service_value', resource_name='resource_name_value', location_uri='location_uri_value')
    operation = client.alter_metadata_resource_location(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)