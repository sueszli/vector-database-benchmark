from google.cloud import gsuiteaddons_v1

def sample_create_deployment():
    if False:
        return 10
    client = gsuiteaddons_v1.GSuiteAddOnsClient()
    request = gsuiteaddons_v1.CreateDeploymentRequest(parent='parent_value', deployment_id='deployment_id_value')
    response = client.create_deployment(request=request)
    print(response)