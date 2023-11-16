from google.cloud import alloydb_v1beta

def sample_promote_cluster():
    if False:
        for i in range(10):
            print('nop')
    client = alloydb_v1beta.AlloyDBAdminClient()
    request = alloydb_v1beta.PromoteClusterRequest(name='name_value')
    operation = client.promote_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)