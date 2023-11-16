from google.cloud import deploy_v1

def sample_cancel_automation_run():
    if False:
        while True:
            i = 10
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.CancelAutomationRunRequest(name='name_value')
    response = client.cancel_automation_run(request=request)
    print(response)