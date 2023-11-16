from google.cloud import functions_v2

def sample_list_functions():
    if False:
        i = 10
        return i + 15
    client = functions_v2.FunctionServiceClient()
    request = functions_v2.ListFunctionsRequest(parent='parent_value')
    page_result = client.list_functions(request=request)
    for response in page_result:
        print(response)