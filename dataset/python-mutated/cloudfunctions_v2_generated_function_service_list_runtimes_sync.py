from google.cloud import functions_v2

def sample_list_runtimes():
    if False:
        while True:
            i = 10
    client = functions_v2.FunctionServiceClient()
    request = functions_v2.ListRuntimesRequest(parent='parent_value')
    response = client.list_runtimes(request=request)
    print(response)