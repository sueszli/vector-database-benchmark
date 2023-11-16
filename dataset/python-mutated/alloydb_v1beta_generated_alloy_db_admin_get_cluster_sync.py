from google.cloud import alloydb_v1beta

def sample_get_cluster():
    if False:
        i = 10
        return i + 15
    client = alloydb_v1beta.AlloyDBAdminClient()
    request = alloydb_v1beta.GetClusterRequest(name='name_value')
    response = client.get_cluster(request=request)
    print(response)