from google.cloud import dataplex_v1

def sample_delete_lake():
    if False:
        return 10
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.DeleteLakeRequest(name='name_value')
    operation = client.delete_lake(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)