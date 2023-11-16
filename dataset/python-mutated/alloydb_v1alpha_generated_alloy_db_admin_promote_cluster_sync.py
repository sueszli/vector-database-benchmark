from google.cloud import alloydb_v1alpha

def sample_promote_cluster():
    if False:
        print('Hello World!')
    client = alloydb_v1alpha.AlloyDBAdminClient()
    request = alloydb_v1alpha.PromoteClusterRequest(name='name_value')
    operation = client.promote_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)