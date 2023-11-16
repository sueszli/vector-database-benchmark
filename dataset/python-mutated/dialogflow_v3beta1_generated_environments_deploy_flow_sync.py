from google.cloud import dialogflowcx_v3beta1

def sample_deploy_flow():
    if False:
        return 10
    client = dialogflowcx_v3beta1.EnvironmentsClient()
    request = dialogflowcx_v3beta1.DeployFlowRequest(environment='environment_value', flow_version='flow_version_value')
    operation = client.deploy_flow(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)