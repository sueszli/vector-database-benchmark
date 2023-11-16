def delete_model(project_id, model_id):
    if False:
        return 10
    'Delete a model.'
    from google.cloud import automl
    client = automl.AutoMlClient()
    model_full_id = client.model_path(project_id, 'us-central1', model_id)
    response = client.delete_model(name=model_full_id)
    print(f'Model deleted. {response.result()}')