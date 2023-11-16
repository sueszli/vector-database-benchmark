from google.cloud import dataproc_v1

def sample_delete_cluster():
    if False:
        i = 10
        return i + 15
    client = dataproc_v1.ClusterControllerClient()
    request = dataproc_v1.DeleteClusterRequest(project_id='project_id_value', region='region_value', cluster_name='cluster_name_value')
    operation = client.delete_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)