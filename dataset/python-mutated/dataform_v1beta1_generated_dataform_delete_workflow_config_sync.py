from google.cloud import dataform_v1beta1

def sample_delete_workflow_config():
    if False:
        print('Hello World!')
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.DeleteWorkflowConfigRequest(name='name_value')
    client.delete_workflow_config(request=request)