from google.cloud import dataproc_v1

def sample_cancel_job():
    if False:
        for i in range(10):
            print('nop')
    client = dataproc_v1.JobControllerClient()
    request = dataproc_v1.CancelJobRequest(project_id='project_id_value', region='region_value', job_id='job_id_value')
    response = client.cancel_job(request=request)
    print(response)