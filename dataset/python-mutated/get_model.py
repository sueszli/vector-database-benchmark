def get_model(project_id, model_id):
    if False:
        for i in range(10):
            print('nop')
    'Get a model.'
    from google.cloud import automl
    client = automl.AutoMlClient()
    model_full_id = client.model_path(project_id, 'us-central1', model_id)
    model = client.get_model(name=model_full_id)
    if model.deployment_state == automl.Model.DeploymentState.DEPLOYED:
        deployment_state = 'deployed'
    else:
        deployment_state = 'undeployed'
    print(f'Model name: {model.name}')
    print('Model id: {}'.format(model.name.split('/')[-1]))
    print(f'Model display name: {model.display_name}')
    print(f'Model create time: {model.create_time}')
    print(f'Model deployment state: {deployment_state}')