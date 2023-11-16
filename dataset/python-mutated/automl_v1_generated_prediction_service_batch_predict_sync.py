from google.cloud import automl_v1

def sample_batch_predict():
    if False:
        for i in range(10):
            print('nop')
    client = automl_v1.PredictionServiceClient()
    input_config = automl_v1.BatchPredictInputConfig()
    input_config.gcs_source.input_uris = ['input_uris_value1', 'input_uris_value2']
    output_config = automl_v1.BatchPredictOutputConfig()
    output_config.gcs_destination.output_uri_prefix = 'output_uri_prefix_value'
    request = automl_v1.BatchPredictRequest(name='name_value', input_config=input_config, output_config=output_config)
    operation = client.batch_predict(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)