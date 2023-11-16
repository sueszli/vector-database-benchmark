from google.cloud import dataform_v1beta1

def sample_list_workflow_invocations():
    if False:
        for i in range(10):
            print('nop')
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.ListWorkflowInvocationsRequest(parent='parent_value')
    page_result = client.list_workflow_invocations(request=request)
    for response in page_result:
        print(response)