from google.cloud import dataproc_v1

def sample_get_cluster():
    if False:
        while True:
            i = 10
    client = dataproc_v1.ClusterControllerClient()
    request = dataproc_v1.GetClusterRequest(project_id='project_id_value', region='region_value', cluster_name='cluster_name_value')
    response = client.get_cluster(request=request)
    print(response)