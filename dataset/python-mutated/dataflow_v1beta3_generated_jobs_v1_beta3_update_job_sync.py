from google.cloud import dataflow_v1beta3

def sample_update_job():
    if False:
        return 10
    client = dataflow_v1beta3.JobsV1Beta3Client()
    request = dataflow_v1beta3.UpdateJobRequest()
    response = client.update_job(request=request)
    print(response)