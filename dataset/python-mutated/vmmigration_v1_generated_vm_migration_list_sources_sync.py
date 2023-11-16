from google.cloud import vmmigration_v1

def sample_list_sources():
    if False:
        print('Hello World!')
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.ListSourcesRequest(parent='parent_value', page_token='page_token_value')
    page_result = client.list_sources(request=request)
    for response in page_result:
        print(response)