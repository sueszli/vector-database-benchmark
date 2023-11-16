from google.cloud import dataform_v1beta1

def sample_create_workflow_invocation():
    if False:
        for i in range(10):
            print('nop')
    client = dataform_v1beta1.DataformClient()
    workflow_invocation = dataform_v1beta1.WorkflowInvocation()
    workflow_invocation.compilation_result = 'compilation_result_value'
    request = dataform_v1beta1.CreateWorkflowInvocationRequest(parent='parent_value', workflow_invocation=workflow_invocation)
    response = client.create_workflow_invocation(request=request)
    print(response)