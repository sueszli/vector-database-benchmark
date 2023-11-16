from google.cloud import alloydb_v1alpha

def sample_delete_cluster():
    if False:
        for i in range(10):
            print('nop')
    client = alloydb_v1alpha.AlloyDBAdminClient()
    request = alloydb_v1alpha.DeleteClusterRequest(name='name_value')
    operation = client.delete_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)