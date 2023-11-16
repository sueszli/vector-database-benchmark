from google.cloud import workstations_v1beta

def sample_list_workstation_clusters():
    if False:
        return 10
    client = workstations_v1beta.WorkstationsClient()
    request = workstations_v1beta.ListWorkstationClustersRequest(parent='parent_value')
    page_result = client.list_workstation_clusters(request=request)
    for response in page_result:
        print(response)