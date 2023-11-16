from google.cloud import config_v1

def sample_create_deployment():
    if False:
        return 10
    client = config_v1.ConfigClient()
    deployment = config_v1.Deployment()
    deployment.terraform_blueprint.gcs_source = 'gcs_source_value'
    request = config_v1.CreateDeploymentRequest(parent='parent_value', deployment_id='deployment_id_value', deployment=deployment)
    operation = client.create_deployment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)