from google.cloud import dataplex_v1

def sample_list_entities():
    if False:
        i = 10
        return i + 15
    client = dataplex_v1.MetadataServiceClient()
    request = dataplex_v1.ListEntitiesRequest(parent='parent_value', view='FILESETS')
    page_result = client.list_entities(request=request)
    for response in page_result:
        print(response)