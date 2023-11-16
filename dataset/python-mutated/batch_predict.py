def batch_predict(project_id, model_id, input_uri, output_uri):
    if False:
        for i in range(10):
            print('nop')
    'Batch predict'
    from google.cloud import automl
    prediction_client = automl.PredictionServiceClient()
    model_full_id = f'projects/{project_id}/locations/us-central1/models/{model_id}'
    gcs_source = automl.GcsSource(input_uris=[input_uri])
    input_config = automl.BatchPredictInputConfig(gcs_source=gcs_source)
    gcs_destination = automl.GcsDestination(output_uri_prefix=output_uri)
    output_config = automl.BatchPredictOutputConfig(gcs_destination=gcs_destination)
    response = prediction_client.batch_predict(name=model_full_id, input_config=input_config, output_config=output_config)
    print('Waiting for operation to complete...')
    print(f'Batch Prediction results saved to Cloud Storage bucket. {response.result()}')