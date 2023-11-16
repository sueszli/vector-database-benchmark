from google.cloud import clouddms_v1

def sample_describe_database_entities():
    if False:
        print('Hello World!')
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.DescribeDatabaseEntitiesRequest(conversion_workspace='conversion_workspace_value', tree='DESTINATION_TREE')
    page_result = client.describe_database_entities(request=request)
    for response in page_result:
        print(response)