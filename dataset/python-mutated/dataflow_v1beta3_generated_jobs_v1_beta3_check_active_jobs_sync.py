from google.cloud import dataflow_v1beta3

def sample_check_active_jobs():
    if False:
        print('Hello World!')
    client = dataflow_v1beta3.JobsV1Beta3Client()
    request = dataflow_v1beta3.CheckActiveJobsRequest()
    response = client.check_active_jobs(request=request)
    print(response)