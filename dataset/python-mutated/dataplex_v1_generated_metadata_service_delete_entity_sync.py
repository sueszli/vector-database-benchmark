from google.cloud import dataplex_v1

def sample_delete_entity():
    if False:
        print('Hello World!')
    client = dataplex_v1.MetadataServiceClient()
    request = dataplex_v1.DeleteEntityRequest(name='name_value', etag='etag_value')
    client.delete_entity(request=request)