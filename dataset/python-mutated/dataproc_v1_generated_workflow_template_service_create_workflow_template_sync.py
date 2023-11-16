from google.cloud import dataproc_v1

def sample_create_workflow_template():
    if False:
        i = 10
        return i + 15
    client = dataproc_v1.WorkflowTemplateServiceClient()
    template = dataproc_v1.WorkflowTemplate()
    template.id = 'id_value'
    template.placement.managed_cluster.cluster_name = 'cluster_name_value'
    template.jobs.hadoop_job.main_jar_file_uri = 'main_jar_file_uri_value'
    template.jobs.step_id = 'step_id_value'
    request = dataproc_v1.CreateWorkflowTemplateRequest(parent='parent_value', template=template)
    response = client.create_workflow_template(request=request)
    print(response)