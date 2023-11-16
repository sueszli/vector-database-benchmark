from google.cloud import dataplex_v1

def sample_delete_environment():
    if False:
        while True:
            i = 10
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.DeleteEnvironmentRequest(name='name_value')
    operation = client.delete_environment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)