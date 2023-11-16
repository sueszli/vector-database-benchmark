from google.cloud import automl_v1beta1

def sample_export_model():
    if False:
        while True:
            i = 10
    client = automl_v1beta1.AutoMlClient()
    request = automl_v1beta1.ExportModelRequest(name='name_value')
    operation = client.export_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)