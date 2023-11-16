from google.cloud import workstations_v1beta

def sample_list_workstation_configs():
    if False:
        i = 10
        return i + 15
    client = workstations_v1beta.WorkstationsClient()
    request = workstations_v1beta.ListWorkstationConfigsRequest(parent='parent_value')
    page_result = client.list_workstation_configs(request=request)
    for response in page_result:
        print(response)