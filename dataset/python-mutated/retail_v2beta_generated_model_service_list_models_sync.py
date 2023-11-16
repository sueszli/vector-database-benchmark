from google.cloud import retail_v2beta

def sample_list_models():
    if False:
        while True:
            i = 10
    client = retail_v2beta.ModelServiceClient()
    request = retail_v2beta.ListModelsRequest(parent='parent_value')
    page_result = client.list_models(request=request)
    for response in page_result:
        print(response)