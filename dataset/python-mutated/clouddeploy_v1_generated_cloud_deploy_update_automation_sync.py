from google.cloud import deploy_v1

def sample_update_automation():
    if False:
        return 10
    client = deploy_v1.CloudDeployClient()
    automation = deploy_v1.Automation()
    automation.service_account = 'service_account_value'
    automation.rules.promote_release_rule.id = 'id_value'
    request = deploy_v1.UpdateAutomationRequest(automation=automation)
    operation = client.update_automation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)