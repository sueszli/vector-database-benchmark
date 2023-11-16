def predict(project_id, model_id, content):
    if False:
        i = 10
        return i + 15
    'Predict.'
    from google.cloud import automl
    prediction_client = automl.PredictionServiceClient()
    model_full_id = automl.AutoMlClient.model_path(project_id, 'us-central1', model_id)
    text_snippet = automl.TextSnippet(content=content, mime_type='text/plain')
    payload = automl.ExamplePayload(text_snippet=text_snippet)
    response = prediction_client.predict(name=model_full_id, payload=payload)
    for annotation_payload in response.payload:
        print(f'Text Extract Entity Types: {annotation_payload.display_name}')
        print(f'Text Score: {annotation_payload.text_extraction.score}')
        text_segment = annotation_payload.text_extraction.text_segment
        print(f'Text Extract Entity Content: {text_segment.content}')
        print(f'Text Start Offset: {text_segment.start_offset}')
        print(f'Text End Offset: {text_segment.end_offset}')