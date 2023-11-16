from google.cloud import alloydb_v1alpha

def sample_list_clusters():
    if False:
        print('Hello World!')
    client = alloydb_v1alpha.AlloyDBAdminClient()
    request = alloydb_v1alpha.ListClustersRequest(parent='parent_value')
    page_result = client.list_clusters(request=request)
    for response in page_result:
        print(response)