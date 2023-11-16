from google.cloud import dataproc_v1

def sample_list_clusters():
    if False:
        i = 10
        return i + 15
    client = dataproc_v1.ClusterControllerClient()
    request = dataproc_v1.ListClustersRequest(project_id='project_id_value', region='region_value')
    page_result = client.list_clusters(request=request)
    for response in page_result:
        print(response)