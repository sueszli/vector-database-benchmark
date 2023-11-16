from google.cloud import alloydb_v1alpha

def sample_get_cluster():
    if False:
        while True:
            i = 10
    client = alloydb_v1alpha.AlloyDBAdminClient()
    request = alloydb_v1alpha.GetClusterRequest(name='name_value')
    response = client.get_cluster(request=request)
    print(response)