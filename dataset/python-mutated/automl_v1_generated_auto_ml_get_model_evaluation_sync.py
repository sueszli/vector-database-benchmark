from google.cloud import automl_v1

def sample_get_model_evaluation():
    if False:
        print('Hello World!')
    client = automl_v1.AutoMlClient()
    request = automl_v1.GetModelEvaluationRequest(name='name_value')
    response = client.get_model_evaluation(request=request)
    print(response)