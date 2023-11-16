from google.cloud import dataflow_v1beta3

def sample_aggregated_list_jobs():
    if False:
        return 10
    client = dataflow_v1beta3.JobsV1Beta3Client()
    request = dataflow_v1beta3.ListJobsRequest()
    page_result = client.aggregated_list_jobs(request=request)
    for response in page_result:
        print(response)