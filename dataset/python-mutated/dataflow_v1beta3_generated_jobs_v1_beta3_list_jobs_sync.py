from google.cloud import dataflow_v1beta3

def sample_list_jobs():
    if False:
        for i in range(10):
            print('nop')
    client = dataflow_v1beta3.JobsV1Beta3Client()
    request = dataflow_v1beta3.ListJobsRequest()
    page_result = client.list_jobs(request=request)
    for response in page_result:
        print(response)