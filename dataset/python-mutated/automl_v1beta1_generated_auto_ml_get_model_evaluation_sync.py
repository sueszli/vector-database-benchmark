from google.cloud import automl_v1beta1

def sample_get_model_evaluation():
    if False:
        return 10
    client = automl_v1beta1.AutoMlClient()
    request = automl_v1beta1.GetModelEvaluationRequest(name='name_value')
    response = client.get_model_evaluation(request=request)
    print(response)