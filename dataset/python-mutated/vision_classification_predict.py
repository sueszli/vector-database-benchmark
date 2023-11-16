def predict(project_id, model_id, file_path):
    if False:
        return 10
    'Predict.'
    from google.cloud import automl
    prediction_client = automl.PredictionServiceClient()
    model_full_id = automl.AutoMlClient.model_path(project_id, 'us-central1', model_id)
    with open(file_path, 'rb') as content_file:
        content = content_file.read()
    image = automl.Image(image_bytes=content)
    payload = automl.ExamplePayload(image=image)
    params = {'score_threshold': '0.8'}
    request = automl.PredictRequest(name=model_full_id, payload=payload, params=params)
    response = prediction_client.predict(request=request)
    print('Prediction results:')
    for result in response.payload:
        print(f'Predicted class name: {result.display_name}')
        print(f'Predicted class score: {result.classification.score}')