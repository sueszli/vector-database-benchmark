from google.cloud import notebooks_v1

def sample_update_instance_metadata_items():
    if False:
        for i in range(10):
            print('nop')
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.UpdateInstanceMetadataItemsRequest(name='name_value')
    response = client.update_instance_metadata_items(request=request)
    print(response)