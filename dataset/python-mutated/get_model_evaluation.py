def get_model_evaluation(project_id, model_id, model_evaluation_id):
    if False:
        return 10
    'Get model evaluation.'
    from google.cloud import automl
    client = automl.AutoMlClient()
    model_path = client.model_path(project_id, 'us-central1', model_id)
    model_evaluation_full_id = f'{model_path}/modelEvaluations/{model_evaluation_id}'
    response = client.get_model_evaluation(name=model_evaluation_full_id)
    print(f'Model evaluation name: {response.name}')
    print(f'Model annotation spec id: {response.annotation_spec_id}')
    print(f'Create Time: {response.create_time}')
    print(f'Evaluation example count: {response.evaluated_example_count}')
    print('Entity extraction model evaluation metrics: {}'.format(response.text_extraction_evaluation_metrics))
    print('Sentiment analysis model evaluation metrics: {}'.format(response.text_sentiment_evaluation_metrics))
    print('Classification model evaluation metrics: {}'.format(response.classification_evaluation_metrics))
    print('Translation model evaluation metrics: {}'.format(response.translation_evaluation_metrics))
    print('Object detection model evaluation metrics: {}'.format(response.image_object_detection_evaluation_metrics))