from google.cloud import retail_v2alpha

def sample_list_models():
    if False:
        print('Hello World!')
    client = retail_v2alpha.ModelServiceClient()
    request = retail_v2alpha.ListModelsRequest(parent='parent_value')
    page_result = client.list_models(request=request)
    for response in page_result:
        print(response)