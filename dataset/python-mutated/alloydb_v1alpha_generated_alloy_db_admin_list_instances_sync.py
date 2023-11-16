from google.cloud import alloydb_v1alpha

def sample_list_instances():
    if False:
        return 10
    client = alloydb_v1alpha.AlloyDBAdminClient()
    request = alloydb_v1alpha.ListInstancesRequest(parent='parent_value')
    page_result = client.list_instances(request=request)
    for response in page_result:
        print(response)