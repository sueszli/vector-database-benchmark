from google.cloud import dataproc_v1

def sample_delete_job():
    if False:
        while True:
            i = 10
    client = dataproc_v1.JobControllerClient()
    request = dataproc_v1.DeleteJobRequest(project_id='project_id_value', region='region_value', job_id='job_id_value')
    client.delete_job(request=request)