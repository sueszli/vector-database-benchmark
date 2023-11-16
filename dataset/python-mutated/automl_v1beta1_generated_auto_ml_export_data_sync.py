from google.cloud import automl_v1beta1

def sample_export_data():
    if False:
        for i in range(10):
            print('nop')
    client = automl_v1beta1.AutoMlClient()
    request = automl_v1beta1.ExportDataRequest(name='name_value')
    operation = client.export_data(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)