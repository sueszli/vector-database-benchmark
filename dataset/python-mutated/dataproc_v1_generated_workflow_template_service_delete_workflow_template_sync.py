from google.cloud import dataproc_v1

def sample_delete_workflow_template():
    if False:
        return 10
    client = dataproc_v1.WorkflowTemplateServiceClient()
    request = dataproc_v1.DeleteWorkflowTemplateRequest(name='name_value')
    client.delete_workflow_template(request=request)