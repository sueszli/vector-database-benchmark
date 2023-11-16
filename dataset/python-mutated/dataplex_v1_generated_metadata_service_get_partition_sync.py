from google.cloud import dataplex_v1

def sample_get_partition():
    if False:
        for i in range(10):
            print('nop')
    client = dataplex_v1.MetadataServiceClient()
    request = dataplex_v1.GetPartitionRequest(name='name_value')
    response = client.get_partition(request=request)
    print(response)