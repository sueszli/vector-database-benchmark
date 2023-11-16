from google.cloud import automl_v1beta1

def sample_update_table_spec():
    if False:
        i = 10
        return i + 15
    client = automl_v1beta1.AutoMlClient()
    request = automl_v1beta1.UpdateTableSpecRequest()
    response = client.update_table_spec(request=request)
    print(response)