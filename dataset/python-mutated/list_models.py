def list_models(project_id):
    if False:
        i = 10
        return i + 15
    'List models.'
    from google.cloud import automl
    client = automl.AutoMlClient()
    project_location = f'projects/{project_id}/locations/us-central1'
    request = automl.ListModelsRequest(parent=project_location, filter='')
    response = client.list_models(request=request)
    print('List of models:')
    for model in response:
        if model.deployment_state == automl.Model.DeploymentState.DEPLOYED:
            deployment_state = 'deployed'
        else:
            deployment_state = 'undeployed'
        print(f'Model name: {model.name}')
        print('Model id: {}'.format(model.name.split('/')[-1]))
        print(f'Model display name: {model.display_name}')
        print(f'Model create time: {model.create_time}')
        print(f'Model deployment state: {deployment_state}')