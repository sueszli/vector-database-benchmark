from google.ai import generativelanguage_v1beta3

def sample_list_models():
    if False:
        while True:
            i = 10
    client = generativelanguage_v1beta3.ModelServiceClient()
    request = generativelanguage_v1beta3.ListModelsRequest()
    page_result = client.list_models(request=request)
    for response in page_result:
        print(response)