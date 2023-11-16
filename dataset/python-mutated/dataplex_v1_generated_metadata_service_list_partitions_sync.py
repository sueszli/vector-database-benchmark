from google.cloud import dataplex_v1

def sample_list_partitions():
    if False:
        i = 10
        return i + 15
    client = dataplex_v1.MetadataServiceClient()
    request = dataplex_v1.ListPartitionsRequest(parent='parent_value')
    page_result = client.list_partitions(request=request)
    for response in page_result:
        print(response)