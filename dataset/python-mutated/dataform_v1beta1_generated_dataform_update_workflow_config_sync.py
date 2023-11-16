from google.cloud import dataform_v1beta1

def sample_update_workflow_config():
    if False:
        while True:
            i = 10
    client = dataform_v1beta1.DataformClient()
    workflow_config = dataform_v1beta1.WorkflowConfig()
    workflow_config.release_config = 'release_config_value'
    request = dataform_v1beta1.UpdateWorkflowConfigRequest(workflow_config=workflow_config)
    response = client.update_workflow_config(request=request)
    print(response)