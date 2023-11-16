from google.cloud import automl_v1beta1

def sample_update_column_spec():
    if False:
        print('Hello World!')
    client = automl_v1beta1.AutoMlClient()
    request = automl_v1beta1.UpdateColumnSpecRequest()
    response = client.update_column_spec(request=request)
    print(response)