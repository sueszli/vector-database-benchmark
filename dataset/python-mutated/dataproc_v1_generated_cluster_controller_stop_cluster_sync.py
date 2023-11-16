from google.cloud import dataproc_v1

def sample_stop_cluster():
    if False:
        print('Hello World!')
    client = dataproc_v1.ClusterControllerClient()
    request = dataproc_v1.StopClusterRequest(project_id='project_id_value', region='region_value', cluster_name='cluster_name_value')
    operation = client.stop_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)