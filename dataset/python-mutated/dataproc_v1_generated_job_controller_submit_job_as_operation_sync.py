from google.cloud import dataproc_v1

def sample_submit_job_as_operation():
    if False:
        print('Hello World!')
    client = dataproc_v1.JobControllerClient()
    job = dataproc_v1.Job()
    job.hadoop_job.main_jar_file_uri = 'main_jar_file_uri_value'
    job.placement.cluster_name = 'cluster_name_value'
    request = dataproc_v1.SubmitJobRequest(project_id='project_id_value', region='region_value', job=job)
    operation = client.submit_job_as_operation(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)