import sys
from google.cloud import dataproc_v1 as dataproc

def instantiate_inline_workflow_template(project_id, region):
    if False:
        print('Hello World!')
    'This sample walks a user through submitting a workflow\n    for a Cloud Dataproc using the Python client library.\n\n    Args:\n        project_id (string): Project to use for running the workflow.\n        region (string): Region where the workflow resources should live.\n    '
    workflow_template_client = dataproc.WorkflowTemplateServiceClient(client_options={'api_endpoint': f'{region}-dataproc.googleapis.com:443'})
    parent = f'projects/{project_id}/regions/{region}'
    template = {'jobs': [{'hadoop_job': {'main_jar_file_uri': 'file:///usr/lib/hadoop-mapreduce/hadoop-mapreduce-examples.jar', 'args': ['teragen', '1000', 'hdfs:///gen/']}, 'step_id': 'teragen'}, {'hadoop_job': {'main_jar_file_uri': 'file:///usr/lib/hadoop-mapreduce/hadoop-mapreduce-examples.jar', 'args': ['terasort', 'hdfs:///gen/', 'hdfs:///sort/']}, 'step_id': 'terasort', 'prerequisite_step_ids': ['teragen']}], 'placement': {'managed_cluster': {'cluster_name': 'my-managed-cluster', 'config': {'gce_cluster_config': {'zone_uri': 'us-central1-a'}}}}}
    operation = workflow_template_client.instantiate_inline_workflow_template(request={'parent': parent, 'template': template})
    operation.result()
    print('Workflow ran successfully.')
if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit('python instantiate_inline_workflow_template.py ' + 'project_id region')
    project_id = sys.argv[1]
    region = sys.argv[2]
    instantiate_inline_workflow_template(project_id, region)