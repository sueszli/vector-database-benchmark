from google.cloud import dataform_v1beta1

def sample_cancel_workflow_invocation():
    if False:
        i = 10
        return i + 15
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.CancelWorkflowInvocationRequest(name='name_value')
    client.cancel_workflow_invocation(request=request)