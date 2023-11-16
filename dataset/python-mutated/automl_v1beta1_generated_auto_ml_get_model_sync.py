from google.cloud import automl_v1beta1

def sample_get_model():
    if False:
        print('Hello World!')
    client = automl_v1beta1.AutoMlClient()
    request = automl_v1beta1.GetModelRequest(name='name_value')
    response = client.get_model(request=request)
    print(response)