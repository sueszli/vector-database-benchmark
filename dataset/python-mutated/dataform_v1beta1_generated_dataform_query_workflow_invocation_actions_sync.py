from google.cloud import dataform_v1beta1

def sample_query_workflow_invocation_actions():
    if False:
        i = 10
        return i + 15
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.QueryWorkflowInvocationActionsRequest(name='name_value')
    page_result = client.query_workflow_invocation_actions(request=request)
    for response in page_result:
        print(response)