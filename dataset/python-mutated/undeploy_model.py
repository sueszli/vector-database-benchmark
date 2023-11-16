def undeploy_model(project_id, model_id):
    if False:
        i = 10
        return i + 15
    'Undeploy a model.'
    from google.cloud import automl
    client = automl.AutoMlClient()
    model_full_id = client.model_path(project_id, 'us-central1', model_id)
    response = client.undeploy_model(name=model_full_id)
    print(f'Model undeployment finished. {response.result()}')