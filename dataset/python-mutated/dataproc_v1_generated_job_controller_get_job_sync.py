from google.cloud import dataproc_v1

def sample_get_job():
    if False:
        for i in range(10):
            print('nop')
    client = dataproc_v1.JobControllerClient()
    request = dataproc_v1.GetJobRequest(project_id='project_id_value', region='region_value', job_id='job_id_value')
    response = client.get_job(request=request)
    print(response)