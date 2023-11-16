from google.cloud import automl_v1beta1

def sample_list_models():
    if False:
        return 10
    client = automl_v1beta1.AutoMlClient()
    request = automl_v1beta1.ListModelsRequest(parent='parent_value')
    page_result = client.list_models(request=request)
    for response in page_result:
        print(response)