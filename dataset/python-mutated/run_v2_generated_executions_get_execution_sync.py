from google.cloud import run_v2

def sample_get_execution():
    if False:
        i = 10
        return i + 15
    client = run_v2.ExecutionsClient()
    request = run_v2.GetExecutionRequest(name='name_value')
    response = client.get_execution(request=request)
    print(response)