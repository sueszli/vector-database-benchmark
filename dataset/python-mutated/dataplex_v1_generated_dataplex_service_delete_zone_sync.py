from google.cloud import dataplex_v1

def sample_delete_zone():
    if False:
        return 10
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.DeleteZoneRequest(name='name_value')
    operation = client.delete_zone(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)