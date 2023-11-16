from google.cloud import gsuiteaddons_v1

def sample_replace_deployment():
    if False:
        return 10
    client = gsuiteaddons_v1.GSuiteAddOnsClient()
    request = gsuiteaddons_v1.ReplaceDeploymentRequest()
    response = client.replace_deployment(request=request)
    print(response)