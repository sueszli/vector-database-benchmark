from google.cloud import vmmigration_v1

def sample_list_groups():
    if False:
        for i in range(10):
            print('nop')
    client = vmmigration_v1.VmMigrationClient()
    request = vmmigration_v1.ListGroupsRequest(parent='parent_value', page_token='page_token_value')
    page_result = client.list_groups(request=request)
    for response in page_result:
        print(response)