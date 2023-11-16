from google.cloud import alloydb_v1

def sample_get_cluster():
    if False:
        return 10
    client = alloydb_v1.AlloyDBAdminClient()
    request = alloydb_v1.GetClusterRequest(name='name_value')
    response = client.get_cluster(request=request)
    print(response)