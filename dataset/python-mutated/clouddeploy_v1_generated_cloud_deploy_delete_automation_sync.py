from google.cloud import deploy_v1

def sample_delete_automation():
    if False:
        i = 10
        return i + 15
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.DeleteAutomationRequest(name='name_value')
    operation = client.delete_automation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)