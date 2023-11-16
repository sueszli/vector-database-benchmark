from google.cloud import alloydb_v1

def sample_delete_cluster():
    if False:
        i = 10
        return i + 15
    client = alloydb_v1.AlloyDBAdminClient()
    request = alloydb_v1.DeleteClusterRequest(name='name_value')
    operation = client.delete_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)