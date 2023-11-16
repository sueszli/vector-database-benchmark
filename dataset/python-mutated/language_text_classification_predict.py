def predict(project_id, model_id, content):
    if False:
        while True:
            i = 10
    'Predict.'
    from google.cloud import automl
    prediction_client = automl.PredictionServiceClient()
    model_full_id = automl.AutoMlClient.model_path(project_id, 'us-central1', model_id)
    text_snippet = automl.TextSnippet(content=content, mime_type='text/plain')
    payload = automl.ExamplePayload(text_snippet=text_snippet)
    response = prediction_client.predict(name=model_full_id, payload=payload)
    for annotation_payload in response.payload:
        print(f'Predicted class name: {annotation_payload.display_name}')
        print(f'Predicted class score: {annotation_payload.classification.score}')