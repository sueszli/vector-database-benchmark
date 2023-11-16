def deploy_model(project_id, model_id):
    if False:
        for i in range(10):
            print('nop')
    'Deploy a model with a specified node count.'
    from google.cloud import automl
    client = automl.AutoMlClient()
    model_full_id = client.model_path(project_id, 'us-central1', model_id)
    metadata = automl.ImageClassificationModelDeploymentMetadata(node_count=2)
    request = automl.DeployModelRequest(name=model_full_id, image_classification_model_deployment_metadata=metadata)
    response = client.deploy_model(request=request)
    print(f'Model deployment finished. {response.result()}')