from google.cloud import dataplex_v1

def sample_delete_task():
    if False:
        print('Hello World!')
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.DeleteTaskRequest(name='name_value')
    operation = client.delete_task(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)