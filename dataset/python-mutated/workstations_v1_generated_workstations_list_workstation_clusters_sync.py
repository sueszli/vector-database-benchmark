from google.cloud import workstations_v1

def sample_list_workstation_clusters():
    if False:
        print('Hello World!')
    client = workstations_v1.WorkstationsClient()
    request = workstations_v1.ListWorkstationClustersRequest(parent='parent_value')
    page_result = client.list_workstation_clusters(request=request)
    for response in page_result:
        print(response)