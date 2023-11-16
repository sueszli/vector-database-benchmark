from google.cloud import deploy_v1

def sample_get_automation_run():
    if False:
        i = 10
        return i + 15
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.GetAutomationRunRequest(name='name_value')
    response = client.get_automation_run(request=request)
    print(response)