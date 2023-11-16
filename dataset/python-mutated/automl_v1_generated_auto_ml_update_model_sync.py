from google.cloud import automl_v1

def sample_update_model():
    if False:
        for i in range(10):
            print('nop')
    client = automl_v1.AutoMlClient()
    request = automl_v1.UpdateModelRequest()
    response = client.update_model(request=request)
    print(response)