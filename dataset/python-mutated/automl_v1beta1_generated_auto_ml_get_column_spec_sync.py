from google.cloud import automl_v1beta1

def sample_get_column_spec():
    if False:
        return 10
    client = automl_v1beta1.AutoMlClient()
    request = automl_v1beta1.GetColumnSpecRequest(name='name_value')
    response = client.get_column_spec(request=request)
    print(response)