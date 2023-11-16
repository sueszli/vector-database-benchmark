from google.cloud import clouddms_v1

def sample_search_background_jobs():
    if False:
        for i in range(10):
            print('nop')
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.SearchBackgroundJobsRequest(conversion_workspace='conversion_workspace_value')
    response = client.search_background_jobs(request=request)
    print(response)