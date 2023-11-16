from google.cloud import deploy_v1

def sample_create_automation():
    if False:
        for i in range(10):
            print('nop')
    client = deploy_v1.CloudDeployClient()
    automation = deploy_v1.Automation()
    automation.service_account = 'service_account_value'
    automation.rules.promote_release_rule.id = 'id_value'
    request = deploy_v1.CreateAutomationRequest(parent='parent_value', automation_id='automation_id_value', automation=automation)
    operation = client.create_automation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)