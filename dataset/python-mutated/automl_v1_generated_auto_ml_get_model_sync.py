from google.cloud import automl_v1

def sample_get_model():
    if False:
        return 10
    client = automl_v1.AutoMlClient()
    request = automl_v1.GetModelRequest(name='name_value')
    response = client.get_model(request=request)
    print(response)