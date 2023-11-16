from google.cloud import dataform_v1beta1

def sample_get_workflow_invocation():
    if False:
        i = 10
        return i + 15
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.GetWorkflowInvocationRequest(name='name_value')
    response = client.get_workflow_invocation(request=request)
    print(response)