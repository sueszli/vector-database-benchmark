from google.cloud import retail_v2

def sample_list_models():
    if False:
        i = 10
        return i + 15
    client = retail_v2.ModelServiceClient()
    request = retail_v2.ListModelsRequest(parent='parent_value')
    page_result = client.list_models(request=request)
    for response in page_result:
        print(response)