from google.cloud import config_v1

def sample_update_deployment():
    if False:
        while True:
            i = 10
    client = config_v1.ConfigClient()
    deployment = config_v1.Deployment()
    deployment.terraform_blueprint.gcs_source = 'gcs_source_value'
    request = config_v1.UpdateDeploymentRequest(deployment=deployment)
    operation = client.update_deployment(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)