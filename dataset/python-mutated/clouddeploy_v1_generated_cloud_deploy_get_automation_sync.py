from google.cloud import deploy_v1

def sample_get_automation():
    if False:
        i = 10
        return i + 15
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.GetAutomationRequest(name='name_value')
    response = client.get_automation(request=request)
    print(response)