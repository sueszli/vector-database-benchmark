from google.cloud import dataform_v1beta1

def sample_list_workflow_configs():
    if False:
        for i in range(10):
            print('nop')
    client = dataform_v1beta1.DataformClient()
    request = dataform_v1beta1.ListWorkflowConfigsRequest(parent='parent_value')
    page_result = client.list_workflow_configs(request=request)
    for response in page_result:
        print(response)