from google.cloud import automl_v1

def sample_get_dataset():
    if False:
        for i in range(10):
            print('nop')
    client = automl_v1.AutoMlClient()
    request = automl_v1.GetDatasetRequest(name='name_value')
    response = client.get_dataset(request=request)
    print(response)