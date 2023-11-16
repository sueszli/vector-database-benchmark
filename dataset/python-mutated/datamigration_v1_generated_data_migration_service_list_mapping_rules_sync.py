from google.cloud import clouddms_v1

def sample_list_mapping_rules():
    if False:
        print('Hello World!')
    client = clouddms_v1.DataMigrationServiceClient()
    request = clouddms_v1.ListMappingRulesRequest(parent='parent_value')
    page_result = client.list_mapping_rules(request=request)
    for response in page_result:
        print(response)