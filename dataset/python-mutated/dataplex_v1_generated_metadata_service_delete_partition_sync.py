from google.cloud import dataplex_v1

def sample_delete_partition():
    if False:
        i = 10
        return i + 15
    client = dataplex_v1.MetadataServiceClient()
    request = dataplex_v1.DeletePartitionRequest(name='name_value')
    client.delete_partition(request=request)