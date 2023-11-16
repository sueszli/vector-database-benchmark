from google.cloud import alloydb_v1

def sample_list_instances():
    if False:
        print('Hello World!')
    client = alloydb_v1.AlloyDBAdminClient()
    request = alloydb_v1.ListInstancesRequest(parent='parent_value')
    page_result = client.list_instances(request=request)
    for response in page_result:
        print(response)