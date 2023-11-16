from google.cloud import gsuiteaddons_v1

def sample_delete_deployment():
    if False:
        return 10
    client = gsuiteaddons_v1.GSuiteAddOnsClient()
    request = gsuiteaddons_v1.DeleteDeploymentRequest(name='name_value')
    client.delete_deployment(request=request)