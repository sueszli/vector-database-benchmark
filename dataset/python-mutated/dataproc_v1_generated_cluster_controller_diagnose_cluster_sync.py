from google.cloud import dataproc_v1

def sample_diagnose_cluster():
    if False:
        while True:
            i = 10
    client = dataproc_v1.ClusterControllerClient()
    request = dataproc_v1.DiagnoseClusterRequest(project_id='project_id_value', region='region_value', cluster_name='cluster_name_value')
    operation = client.diagnose_cluster(request=request)
    print('Waiting for operation to complete...')
    response = operation.result()
    print(response)