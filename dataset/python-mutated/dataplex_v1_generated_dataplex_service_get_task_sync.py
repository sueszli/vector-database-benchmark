from google.cloud import dataplex_v1

def sample_get_task():
    if False:
        return 10
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.GetTaskRequest(name='name_value')
    response = client.get_task(request=request)
    print(response)