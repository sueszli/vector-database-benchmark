from google.cloud import dialogflowcx_v3beta1

def sample_get_deployment():
    if False:
        print('Hello World!')
    client = dialogflowcx_v3beta1.DeploymentsClient()
    request = dialogflowcx_v3beta1.GetDeploymentRequest(name='name_value')
    response = client.get_deployment(request=request)
    print(response)