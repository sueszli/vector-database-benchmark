from google.cloud import automl_v1beta1

def sample_get_dataset():
    if False:
        i = 10
        return i + 15
    client = automl_v1beta1.AutoMlClient()
    request = automl_v1beta1.GetDatasetRequest(name='name_value')
    response = client.get_dataset(request=request)
    print(response)