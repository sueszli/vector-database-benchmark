from google.cloud import dataplex_v1

def sample_run_task():
    if False:
        i = 10
        return i + 15
    client = dataplex_v1.DataplexServiceClient()
    request = dataplex_v1.RunTaskRequest(name='name_value')
    response = client.run_task(request=request)
    print(response)