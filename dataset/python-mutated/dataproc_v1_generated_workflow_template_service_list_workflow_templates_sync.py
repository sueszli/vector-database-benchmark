from google.cloud import dataproc_v1

def sample_list_workflow_templates():
    if False:
        for i in range(10):
            print('nop')
    client = dataproc_v1.WorkflowTemplateServiceClient()
    request = dataproc_v1.ListWorkflowTemplatesRequest(parent='parent_value')
    page_result = client.list_workflow_templates(request=request)
    for response in page_result:
        print(response)