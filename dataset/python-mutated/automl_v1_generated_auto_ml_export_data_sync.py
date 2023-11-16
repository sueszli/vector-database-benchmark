from google.cloud import automl_v1

def sample_export_data():
    if False:
        return 10
    client = automl_v1.AutoMlClient()
    output_config = automl_v1.OutputConfig()
    output_config.gcs_destination.output_uri_prefix = 'output_uri_prefix_value'
    request = automl_v1.ExportDataRequest(name='name_value', output_config=output_config)
    operation = client.export_data(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)