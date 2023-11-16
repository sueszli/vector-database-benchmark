def list_model_evaluations(project_id, model_id):
    if False:
        for i in range(10):
            print('nop')
    'List model evaluations.'
    from google.cloud import automl
    client = automl.AutoMlClient()
    model_full_id = client.model_path(project_id, 'us-central1', model_id)
    print('List of model evaluations:')
    for evaluation in client.list_model_evaluations(parent=model_full_id, filter=''):
        print(f'Model evaluation name: {evaluation.name}')
        print(f'Model annotation spec id: {evaluation.annotation_spec_id}')
        print(f'Create Time: {evaluation.create_time}')
        print(f'Evaluation example count: {evaluation.evaluated_example_count}')
        print('Entity extraction model evaluation metrics: {}'.format(evaluation.text_extraction_evaluation_metrics))
        print('Sentiment analysis model evaluation metrics: {}'.format(evaluation.text_sentiment_evaluation_metrics))
        print('Classification model evaluation metrics: {}'.format(evaluation.classification_evaluation_metrics))
        print('Translation model evaluation metrics: {}'.format(evaluation.translation_evaluation_metrics))
        print('Object detection model evaluation metrics: {}\n\n'.format(evaluation.image_object_detection_evaluation_metrics))