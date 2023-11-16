from google.cloud import dialogflowcx_v3

def sample_get_deployment():
    if False:
        return 10
    client = dialogflowcx_v3.DeploymentsClient()
    request = dialogflowcx_v3.GetDeploymentRequest(name='name_value')
    response = client.get_deployment(request=request)
    print(response)