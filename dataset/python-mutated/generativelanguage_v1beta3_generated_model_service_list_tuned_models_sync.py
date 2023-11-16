from google.ai import generativelanguage_v1beta3

def sample_list_tuned_models():
    if False:
        return 10
    client = generativelanguage_v1beta3.ModelServiceClient()
    request = generativelanguage_v1beta3.ListTunedModelsRequest()
    page_result = client.list_tuned_models(request=request)
    for response in page_result:
        print(response)