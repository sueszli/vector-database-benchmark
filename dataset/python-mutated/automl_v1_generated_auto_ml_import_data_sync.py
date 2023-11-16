from google.cloud import automl_v1

def sample_import_data():
    if False:
        for i in range(10):
            print('nop')
    client = automl_v1.AutoMlClient()
    input_config = automl_v1.InputConfig()
    input_config.gcs_source.input_uris = ['input_uris_value1', 'input_uris_value2']
    request = automl_v1.ImportDataRequest(name='name_value', input_config=input_config)
    operation = client.import_data(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)