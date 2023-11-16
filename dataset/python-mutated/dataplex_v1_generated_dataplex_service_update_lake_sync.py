from google.cloud import dataplex_v1

def sample_update_lake():
    if False:
        while True:
            i = 10
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.UpdateLakeRequest()
    operation = client.update_lake(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)