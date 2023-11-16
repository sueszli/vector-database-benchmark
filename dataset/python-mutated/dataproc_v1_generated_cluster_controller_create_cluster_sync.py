from google.cloud import dataproc_v1

def sample_create_cluster():
    if False:
        print('Hello World!')
    client = dataproc_v1.ClusterControllerClient()
    cluster = dataproc_v1.Cluster()
    cluster.project_id = 'project_id_value'
    cluster.cluster_name = 'cluster_name_value'
    request = dataproc_v1.CreateClusterRequest(project_id='project_id_value', region='region_value', cluster=cluster)
    operation = client.create_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)