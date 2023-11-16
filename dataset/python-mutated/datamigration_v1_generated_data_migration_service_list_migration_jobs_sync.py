from google.cloud import clouddms_v1

def sample_list_migration_jobs():
    if False:
        print('Hello World!')
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.ListMigrationJobsRequest(parent='parent_value')
    page_result = client.list_migration_jobs(request=request)
    for response in page_result:
        print(response)