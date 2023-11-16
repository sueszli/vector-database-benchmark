from google.cloud import alloydb_v1beta

def sample_list_clusters():
    if False:
        while True:
            i = 10
    client = alloydb_v1beta.AlloyDBAdminClient()
    request = alloydb_v1beta.ListClustersRequest(parent='parent_value')
    page_result = client.list_clusters(request=request)
    for response in page_result:
        print(response)