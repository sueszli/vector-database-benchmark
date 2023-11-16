from google.cloud import automl_v1beta1

def sample_import_data():
    if False:
        print('Hello World!')
    client = automl_v1beta1.AutoMlClient()
    request = automl_v1beta1.ImportDataRequest(name='name_value')
    operation = client.import_data(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)