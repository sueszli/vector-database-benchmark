from google.cloud import automl_v1

def sample_export_model():
    if False:
        print('Hello World!')
    client = automl_v1.AutoMlClient()
    output_config = automl_v1.ModelExportOutputConfig()
    output_config.gcs_destination.output_uri_prefix = 'output_uri_prefix_value'
    request = automl_v1.ExportModelRequest(name='name_value', output_config=output_config)
    operation = client.export_model(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)