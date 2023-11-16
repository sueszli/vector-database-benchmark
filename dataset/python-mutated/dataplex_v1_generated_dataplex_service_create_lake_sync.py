from google.cloud import dataplex_v1

def sample_create_lake():
    if False:
        while True:
            i = 10
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.CreateLakeRequest(parent='parent_value', lake_id='lake_id_value')
    operation = client.create_lake(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)