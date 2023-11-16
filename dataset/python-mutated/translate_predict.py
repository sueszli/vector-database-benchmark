def predict(project_id, model_id, file_path):
    if False:
        while True:
            i = 10
    'Predict.'
    from google.cloud import automl
    prediction_client = automl.PredictionServiceClient()
    model_full_id = automl.AutoMlClient.model_path(project_id, 'us-central1', model_id)
    with open(file_path, 'rb') as content_file:
        content = content_file.read()
    content.decode('utf-8')
    text_snippet = automl.TextSnippet(content=content)
    payload = automl.ExamplePayload(text_snippet=text_snippet)
    response = prediction_client.predict(name=model_full_id, payload=payload)
    translated_content = response.payload[0].translation.translated_content
    print(f'Translated content: {translated_content.content}')