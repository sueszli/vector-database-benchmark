from google.cloud import automl_v1beta1

def sample_get_table_spec():
    if False:
        for i in range(10):
            print('nop')
    client = automl_v1beta1.AutoMlClient()
    request = automl_v1beta1.GetTableSpecRequest(name='name_value')
    response = client.get_table_spec(request=request)
    print(response)