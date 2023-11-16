from google.cloud import automl_v1

def sample_list_model_evaluations():
    if False:
        i = 10
        return i + 15
    client = automl_v1.AutoMlClient()
    request = automl_v1.ListModelEvaluationsRequest(parent='parent_value', filter='filter_value')
    page_result = client.list_model_evaluations(request=request)
    for response in page_result:
        print(response)