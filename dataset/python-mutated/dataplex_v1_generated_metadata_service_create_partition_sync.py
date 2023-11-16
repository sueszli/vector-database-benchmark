from google.cloud import dataplex_v1

def sample_create_partition():
    if False:
        while True:
            i = 10
    client = dataplex_v1.MetadataServiceClient()
    partition = dataplex_v1.Partition()
    partition.values = ['values_value1', 'values_value2']
    partition.location = 'location_value'
    request = dataplex_v1.CreatePartitionRequest(parent='parent_value', partition=partition)
    response = client.create_partition(request=request)
    print(response)