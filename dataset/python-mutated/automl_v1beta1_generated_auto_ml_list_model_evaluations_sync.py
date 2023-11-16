from google.cloud import automl_v1beta1

def sample_list_model_evaluations():
    if False:
        print('Hello World!')
    client = automl_v1beta1.AutoMlClient()
    request = automl_v1beta1.ListModelEvaluationsRequest(parent='parent_value')
    page_result = client.list_model_evaluations(request=request)
    for response in page_result:
        print(response)