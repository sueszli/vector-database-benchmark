from google.cloud import dataplex_v1

def sample_get_entity():
    if False:
        return 10
    client = dataplex_v1.MetadataServiceClient()
    request = dataplex_v1.GetEntityRequest(name='name_value')
    response = client.get_entity(request=request)
    print(response)