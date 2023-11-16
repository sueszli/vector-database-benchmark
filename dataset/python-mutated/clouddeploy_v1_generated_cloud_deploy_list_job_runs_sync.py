from google.cloud import deploy_v1

def sample_list_job_runs():
    if False:
        i = 10
        return i + 15
    client = deploy_v1.CloudDeployClient()
    request = deploy_v1.ListJobRunsRequest(parent='parent_value')
    page_result = client.list_job_runs(request=request)
    for response in page_result:
        print(response)