from google.cloud import alloydb_v1beta

def sample_list_instances():
    if False:
        while True:
            i = 10
    client = alloydb_v1beta.AlloyDBAdminClient()
    request = alloydb_v1beta.ListInstancesRequest(parent='parent_value')
    page_result = client.list_instances(request=request)
    for response in page_result:
        print(response)