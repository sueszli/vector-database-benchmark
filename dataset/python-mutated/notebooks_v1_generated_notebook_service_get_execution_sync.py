from google.cloud import notebooks_v1

def sample_get_execution():
    if False:
        while True:
            i = 10
    client = notebooks_v1.NotebookServiceClient()
    request = notebooks_v1.GetExecutionRequest(name='name_value')
    response = client.get_execution(request=request)
    print(response)