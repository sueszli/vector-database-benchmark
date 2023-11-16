from google.cloud import gsuiteaddons_v1

def sample_get_deployment():
    if False:
        return 10
    client = gsuiteaddons_v1.GSuiteAddOnsClient()
    request = gsuiteaddons_v1.GetDeploymentRequest(name='name_value')
    response = client.get_deployment(request=request)
    print(response)