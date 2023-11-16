from google.cloud import dataproc_v1

def sample_update_job():
    if False:
        for i in range(10):
            print('nop')
    client = dataproc_v1.JobControllerClient()
    job = dataproc_v1.Job()
    job.hadoop_job.main_jar_file_uri = 'main_jar_file_uri_value'
    job.placement.cluster_name = 'cluster_name_value'
    request = dataproc_v1.UpdateJobRequest(project_id='project_id_value', region='region_value', job_id='job_id_value', job=job)
    response = client.update_job(request=request)
    print(response)