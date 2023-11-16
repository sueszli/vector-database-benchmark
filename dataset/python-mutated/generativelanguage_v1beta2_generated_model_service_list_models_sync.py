from google.ai import generativelanguage_v1beta2

def sample_list_models():
    if False:
        print('Hello World!')
    client = generativelanguage_v1beta2.ModelServiceClient()
    request = generativelanguage_v1beta2.ListModelsRequest()
    page_result = client.list_models(request=request)
    for response in page_result:
        print(response)