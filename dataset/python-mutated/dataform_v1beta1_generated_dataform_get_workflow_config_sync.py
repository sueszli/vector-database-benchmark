from google.cloud import dataform_v1beta1

def sample_get_workflow_config():
    if False:
        return 10
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.GetWorkflowConfigRequest(name='name_value')
    response = client.get_workflow_config(request=request)
    print(response)