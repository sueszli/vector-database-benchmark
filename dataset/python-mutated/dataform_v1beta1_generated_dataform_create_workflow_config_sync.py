from google.cloud import dataform_v1beta1

def sample_create_workflow_config():
    if False:
        return 10
    client = dataform_v1beta1.DataformClient()
    workflow_config = dataform_v1beta1.WorkflowConfig()
    workflow_config.release_config = 'release_config_value'
    request = dataform_v1beta1.CreateWorkflowConfigRequest(parent='parent_value', workflow_config=workflow_config, workflow_config_id='workflow_config_id_value')
    response = client.create_workflow_config(request=request)
    print(response)